"""Hosted hero-demo backend.

The package is intentionally split into a lightweight contract/session layer and a
lazy model engine. Importing :mod:`gemma_sv.demo_server` never loads Gemma, Torch,
Transformers, MLX, or FastAPI.
"""

from .contract import (
    AdmissionResult,
    AdmissionStatus,
    CertificateBand,
    CertificateResult,
    classify_admission,
    classify_certificate,
    kl_from_log_probs,
    sequence_log_score,
)

__all__ = [
    "AdmissionResult",
    "AdmissionStatus",
    "CertificateBand",
    "CertificateResult",
    "classify_admission",
    "classify_certificate",
    "kl_from_log_probs",
    "sequence_log_score",
]

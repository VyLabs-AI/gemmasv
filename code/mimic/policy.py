"""Admission thresholds and summary statistics shared by both evaluations.

Defined once so the two papers judge deletion by one standard. The values match
``gemma_sv/eval_whole_record_unlearning.py``, which set them first.
"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional

# A record must demonstrably raise the readout before its deletion is measurable:
# forgetting cannot be observed on something never recalled.
MIN_SIGNAL = 0.05
MAX_FIRST_TOKEN_RANK = 10


def summary(values: Iterable[float]) -> Optional[Dict[str, float]]:
    """Mean, median, min, and max, or ``None`` when nothing was measurable."""
    present: List[float] = [v for v in values if v is not None]
    if not present:
        return None
    return {
        "mean": statistics.mean(present),
        "median": statistics.median(present),
        "max": max(present),
        "min": min(present),
    }

"""Decay-based forgetting on a Kimi Delta Attention state, for contrast.

KDA already forgets: each step multiplies the recurrent state by a learned
per-key-channel retention vector. That mechanism is untargeted by construction.
The channels of the state are shared by every token ever written, so no setting
of the gate corresponds to "the record about patient 5182" -- scaling the state
attenuates all stored associations together and never reaches zero in finite
time.

This module provides the decay baseline in its most favourable form: a direct
multiplicative attenuation applied to the recurrent state, sweeping the
retention factor. It is the analogue of the coefficient-decay baseline in the
SV-Attention paper, and it exists here to be beaten by exact replay.
"""
from __future__ import annotations

from typing import Any, Sequence

import mlx.core as mx

from .state import classify_layers

# Index of the recurrent delta-rule state within a KDA layer's ArraysCache;
# entries 0-2 are the short-convolution states.
RECURRENT = 3


def scale_recurrent_state(model: Any, cache: Sequence[Any], gamma: float) -> None:
    """Multiply every KDA recurrent state by ``gamma`` in place.

    ``gamma = 1`` is a no-op; ``gamma = 0`` erases the entire linear-attention
    memory, every record along with the target.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")

    kda, _ = classify_layers(model)
    for i in kda:
        state = cache[i][RECURRENT]
        if state is not None:
            cache[i][RECURRENT] = state * gamma
            mx.eval(cache[i][RECURRENT])

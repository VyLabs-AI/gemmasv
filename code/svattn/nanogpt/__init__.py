"""nanoGPT-style decoder with a pluggable attention slot.

The same backbone (token+pos embed -> N x [pre-LN attn + MLP] -> LN -> head) is
used for every variant; only the attention module differs, so SV-Attention, the
softmax/linear/DeltaNet baselines, etc. are swapped via a factory. This is the
integration point for the A6 approximation-gap study (Phase 1) and the A2
matched-state LM-quality demo (Phase 2).
"""
from .model import CharGPT, GPTConfig, build_attn
from .data import Enwik8, batch_iter

__all__ = ["CharGPT", "GPTConfig", "build_attn", "Enwik8", "batch_iter"]

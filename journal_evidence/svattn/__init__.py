"""Support Vector Attention — differentiable max-margin test-time memory.

Building blocks include a differentiable one-class SVDD gate whose maintained
backward reuses the bordered KKT inverse, plus a kernel readout over context
values. The batched training path has a separate numerical contract.
"""
__all__ = []

try:  # torch-dependent modules (differentiable layer); optional for the
    # numpy-only stream/benchmark modules (chunked_gate, eviction_benchmark, ...)
    from .diff_svdd import OneClassSVDD, svdd_solve_partition
    from .sv_attention import SVAttention, rbf_gram
    from .fast_diff_svdd import FastDiffSVDD, fast_diff_svdd, fast_svdd_state
    from .causal_sv_attention import CausalSVAttention, causal_sv_readout
    __all__ += ["OneClassSVDD", "svdd_solve_partition", "SVAttention", "rbf_gram",
                "FastDiffSVDD", "fast_diff_svdd", "fast_svdd_state",
                "CausalSVAttention", "causal_sv_readout"]
except ImportError:
    pass

from .chunked_gate import ChunkedSVGate
__all__ += ["ChunkedSVGate"]

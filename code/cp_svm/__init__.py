"""Cauwenberghs--Poggio incremental/decremental SVM implementations.

The maintained one-class solver supplies SV-Attention's fixed-C verification
path. Numerical claims are paired with explicit coverage and tolerance audits.
See THIRD_PARTY_NOTICES.md for algorithm and reference-implementation credits.
"""

from .kernels import radial_kernel, linear_kernel
from .oneclass_qp import (
    SVDDQPInfeasibleError,
    solve_svdd_qp,
    recover_rho,
    partition_sets,
    solve_svm_qp,
    recover_b_svm,
    solve_svr_qp,
    recover_b_svr,
)
from .oneclass_incremental import OneClassIncrementalSVM
from .oneclass_fast import FastOneClassSVM
from .binary_incremental import BinaryIncrementalSVM
from .svr_incremental import SVRIncremental

__all__ = [
    "radial_kernel",
    "linear_kernel",
    "SVDDQPInfeasibleError",
    "solve_svdd_qp",
    "recover_rho",
    "partition_sets",
    "solve_svm_qp",
    "recover_b_svm",
    "solve_svr_qp",
    "recover_b_svr",
    "OneClassIncrementalSVM",
    "FastOneClassSVM",
    "BinaryIncrementalSVM",
    "SVRIncremental",
]

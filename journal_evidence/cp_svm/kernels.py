"""Kernel functions used by the SVDD solvers.

The radial kernel uses the declared width convention K = exp(-d / kpar^2),
where d is squared Euclidean distance.
"""
from __future__ import annotations

import numpy as np


def _sq_eucl(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix, clamped at zero."""
    dA = np.sum(A * A, axis=1, keepdims=True)
    dB = np.sum(B * B, axis=1, keepdims=True)
    D = dA + dB.T - 2.0 * (A @ B.T)
    D[D < 0] = 0.0
    return D


def radial_kernel(A: np.ndarray, B: np.ndarray, kpar: float) -> np.ndarray:
    """Gaussian kernel, MATLAB convention: exp(-||a-b||^2 / kpar^2)."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    return np.exp(-_sq_eucl(A, B) / (kpar * kpar))


def linear_kernel(A: np.ndarray, B: np.ndarray, kpar: float | None = None) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    return A @ B.T

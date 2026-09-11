"""Fixed-C decrement/refit trial primitive for released contract tests."""

from __future__ import annotations

import numpy as np

from cp_svm import FastOneClassSVM
from cp_svm.kernels import radial_kernel
from cp_svm.oneclass_qp import recover_rho


def _failure_code(stage: str, error: Exception) -> str:
    message = str(error).casefold()
    if "margin set emptied" in message or "margin set empty" in message:
        reason = "margin_empty"
    elif "no feasible" in message:
        reason = "no_feasible_step"
    elif "exceeded max_iter" in message or "exceeded max iter" in message:
        reason = "max_iterations"
    elif "singular" in message:
        reason = "singular_system"
    else:
        reason = type(error).__name__.casefold()
    return f"{stage}:{reason}"


def one_trial(X, rng, nu, kpar, n_forget, forget=None):
    """Compare decrement with an independently solved retained-key reference."""

    n = len(X)
    if n < n_forget + 4:
        return {"status": "skipped", "failure": "input:insufficient_points"}
    C = 1.0 / (nu * n)
    forget = (
        sorted(rng.choice(n, n_forget, replace=False).tolist())
        if forget is None
        else sorted({int(index) for index in forget})
    )
    n_forget = len(forget)
    keep = [index for index in range(n) if index not in set(forget)]
    retained_capacity = len(keep) * C
    base = {
        "n_points": int(n),
        "n_retained": len(keep),
        "n_forget": n_forget,
        "C": float(C),
        "retained_capacity": float(retained_capacity),
    }
    if retained_capacity < 1.0 - 1e-12:
        return {
            **base,
            "status": "failed",
            "failure": "fixed_c_preflight:infeasible",
            "refit_status": "infeasible",
        }

    try:
        model = FastOneClassSVM(
            C=C,
            ktype="r",
            kpar=kpar,
        ).seed_from_qp(X)
        alpha_full = np.array(model.alpha)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            **base,
            "status": "failed",
            "failure": _failure_code("full_seed", error),
            "refit_status": "not_run",
        }

    try:
        reference = FastOneClassSVM(
            C=C,
            ktype="r",
            kpar=kpar,
        ).seed_from_qp(X[keep])
        alpha_fresh = np.array(reference.alpha)
        support_fresh = set(reference.S)
        error_fresh = set(reference.E)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            **base,
            "status": "failed",
            "failure": _failure_code("refit", error),
            "refit_status": "failed",
        }

    try:
        for index in sorted(forget, reverse=True):
            model.remove_point(index)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            **base,
            "status": "failed",
            "failure": _failure_code("decrement", error),
            "refit_status": "completed",
        }

    alpha_decrement = np.array(model.alpha)
    alpha_deviation = float(
        np.max(np.abs(alpha_decrement - alpha_fresh))
    )
    partition_match = (
        set(model.S) == support_fresh and set(model.E) == error_fresh
    )
    kept = X[keep]
    probes = np.vstack(
        [
            kept,
            X[forget],
            kept + 0.1 * rng.standard_normal(kept.shape),
        ]
    )
    function_deviation = float(
        np.max(
            np.abs(
                model.decision_function(probes)
                - reference.decision_function(probes)
            )
        )
    )
    try:
        alpha_decay = alpha_full.copy()
        alpha_decay[forget] *= 0.01
        rho_decay = recover_rho(
            radial_kernel(X, X, kpar),
            alpha_decay,
            C,
        )
        decay_function = (
            2.0 * (radial_kernel(probes, X, kpar) @ alpha_decay)
            - rho_decay
        )
        decay_deviation = float(
            np.max(
                np.abs(
                    decay_function - reference.decision_function(probes)
                )
            )
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {
            **base,
            "status": "failed",
            "failure": _failure_code("decay", error),
        }

    return {
        **base,
        "status": "completed",
        "refit_status": "completed",
        "alpha_deviation": alpha_deviation,
        "partition_match": bool(partition_match),
        "function_deviation": function_deviation,
        "decay_function_deviation": decay_deviation,
    }

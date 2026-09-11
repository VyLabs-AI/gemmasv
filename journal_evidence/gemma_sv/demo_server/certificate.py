"""Reusable float64 decrement/refit gate construction for output certificates."""

from __future__ import annotations

import math
import statistics
from typing import Callable, Mapping, Sequence

import numpy as np


DECAY = 0.01
# This is a pre-output guard on the SVDD decision function, not the reported
# certificate.  The final full-vocabulary output KL remains the claim-bearing
# metric.  Real Gemma Gram matrices are ill-conditioned/non-unique; the coupled
# block path matches fixed-C refit to ~1e-7--1e-6 function deviation while the
# resulting output KL remains far below the 1e-6 hero gate.
FUNCTIONAL_TOLERANCE = 1e-5


class CertificateFeasibilityError(RuntimeError):
    """A fixed-box retained-key refit has no feasible dual solution."""

    def __init__(self, diagnostics: list[dict[str, float | int | bool]]):
        self.diagnostics = diagnostics
        failed = [item for item in diagnostics if not item["feasible"]]
        first = failed[0]
        super().__init__(
            "fixed-C refit is infeasible at boundary "
            f"{first['start']}: retained_count*C={first['retained_capacity']:.6f} < 1"
        )


def fixed_c_feasibility(
    total_tokens: int,
    forget_positions: Sequence[int],
    *,
    nu: float = 0.3,
    chunk: int = 128,
    box_C: float | None = None,
    box_C_by_boundary: Mapping[int, float] | None = None,
    per_boundary_box: bool = False,
) -> tuple[
    float | dict[int, float],
    list[dict[str, float | int | bool | str]],
]:
    """Preflight every affected retained-prefix QP under the fixed box.

    The SVDD dual requires ``sum(alpha)=1`` and ``alpha_i<=C``.  Therefore a
    retained prefix with ``n`` points is feasible iff ``n*C>=1``.  Deleting
    many early positions can violate this even though the original prefix was
    feasible; silently changing C would change the paper's counterfactual.
    """

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if not 0 < nu <= 1:
        raise ValueError("nu must be in (0, 1] for a feasible SVDD box")
    if box_C is not None and (
        box_C_by_boundary is not None or per_boundary_box
    ):
        raise ValueError("choose one global or per-boundary box specification")
    if box_C_by_boundary is not None and per_boundary_box:
        raise ValueError(
            "box_C_by_boundary already selects per-boundary boxes"
        )
    starts = list(range(chunk, total_tokens, chunk))
    if box_C_by_boundary is not None:
        boxes = {
            int(start): float(value)
            for start, value in box_C_by_boundary.items()
        }
        if set(boxes) != set(starts):
            raise ValueError("box_C_by_boundary coverage differs from boundaries")
    elif per_boundary_box:
        boxes = {
            start: min(1.0, 1.0 / (float(nu) * start))
            for start in starts
        }
    else:
        C = (
            min(1.0, 1.0 / (float(nu) * total_tokens))
            if box_C is None
            else float(box_C)
        )
        boxes = {start: C for start in starts}
    if any(
        not math.isfinite(value) or not 0 < value <= 1
        for value in boxes.values()
    ):
        raise ValueError("every box C must be finite and in (0, 1]")
    forget = tuple(sorted({int(position) for position in forget_positions}))
    diagnostics: list[dict[str, float | int | bool | str]] = []
    for start in starts:
        C = boxes[start]
        threshold = int(math.ceil(1.0 / C)) + 2
        if start < threshold:
            continue
        forgotten_count = sum(position < start for position in forget)
        if not forgotten_count:
            continue
        retained_count = start - forgotten_count
        retained_capacity = retained_count * C
        deleted_fraction = forgotten_count / start
        maximum_deleted_fraction = max(0.0, 1.0 - 1.0 / (C * start))
        feasible = retained_capacity >= 1.0 - 1e-12
        diagnostics.append(
            {
                "start": start,
                "box_C": C,
                "forgotten_count": forgotten_count,
                "retained_count": retained_count,
                "retained_capacity": retained_capacity,
                "deleted_fraction": deleted_fraction,
                "maximum_deleted_fraction": maximum_deleted_fraction,
                "feasible": feasible,
                "fallback": "none" if feasible else "full_repack",
            }
        )
    box_result: float | dict[int, float] = (
        boxes
        if box_C_by_boundary is not None or per_boundary_box
        else next(iter(boxes.values()), float(box_C or 1.0))
    )
    return box_result, diagnostics


def median_kernel_width(keys: np.ndarray) -> float:
    d2 = np.sum((keys[:, None] - keys[None]) ** 2, axis=-1)
    positive = d2[d2 > 0]
    if not positive.size:
        raise ValueError("cannot choose an RBF width from duplicate-only keys")
    return float(np.sqrt(np.median(positive)))


def certificate_overrides(
    keys: Mapping[int, np.ndarray],
    layer_ids: Sequence[int],
    forget_positions: Sequence[int],
    total_tokens: int,
    n_heads: int,
    *,
    nu: float = 0.3,
    chunk: int = 128,
    decay: float = DECAY,
    progress: Callable[[int, int], None] | None = None,
    skip_incremental: bool = False,
    box_C: float | None = None,
    box_C_by_boundary: Mapping[int, float] | None = None,
    per_boundary_box: bool = False,
    kpar_by_layer: Mapping[int, float] | None = None,
) -> dict:
    """Build exact, refit, and decay alpha overrides at every gated boundary.

    ``keys[layer]`` has shape ``(heads, tokens, head_dim)`` and must already be
    captured from the certificate model in float64.  A margin-set failure falls
    back to the from-scratch refit and is surfaced in the returned diagnostics.
    ``skip_incremental`` populates the exact case directly with the refit
    solution (their equivalence is the certified claim elsewhere); use it when
    only the refit state is consumed, e.g.\\ for imprint measurements.
    ``box_C`` and ``kpar_by_layer`` must be supplied by persistent callers so
    the reference solves exactly the objective frozen in each decode session;
    omitting them preserves the standalone median-bandwidth compatibility path.
    """

    from cp_svm import FastOneClassSVM

    if total_tokens <= 0 or n_heads <= 0:
        raise ValueError("certificate dimensions must be positive")
    if not layer_ids:
        raise ValueError("at least one global layer is required")

    box_spec, feasibility = fixed_c_feasibility(
        total_tokens,
        forget_positions,
        nu=nu,
        chunk=chunk,
        box_C=box_C,
        box_C_by_boundary=box_C_by_boundary,
        per_boundary_box=per_boundary_box,
    )
    if any(not item["feasible"] for item in feasibility):
        raise CertificateFeasibilityError(feasibility)
    boxes = (
        box_spec
        if isinstance(box_spec, dict)
        else {
            start: float(box_spec)
            for start in range(chunk, total_tokens, chunk)
        }
    )
    gated = [
        start
        for start, value in boxes.items()
        if start >= int(math.ceil(1.0 / value)) + 2
    ]
    cases = {
        name: {int(layer_id): {} for layer_id in layer_ids}
        for name in ("exact", "refit", "decay")
    }
    forget = tuple(sorted({int(position) for position in forget_positions}))
    frozen_kpars: dict[int, float] | None = None
    if kpar_by_layer is not None:
        missing = sorted(set(int(layer_id) for layer_id in layer_ids) - set(kpar_by_layer))
        extra = sorted(set(kpar_by_layer) - set(int(layer_id) for layer_id in layer_ids))
        if missing or extra:
            raise ValueError(
                "kpar_by_layer coverage differs from layer_ids; "
                f"missing={missing}, extra={extra}"
            )
        frozen_kpars = {}
        for layer_id in layer_ids:
            value = float(kpar_by_layer[int(layer_id)])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"kpar_by_layer[{int(layer_id)}] must be finite and positive"
                )
            frozen_kpars[int(layer_id)] = value
    expected = len(layer_ids) * len(gated) * n_heads
    completed = n_solves = n_fallback = 0
    max_functional_deviation = 0.0
    max_candidate_deviation = 0.0
    fallback_details: list[dict[str, int | str | float]] = []
    affected_support_fraction: list[float] = []
    affected_margin_fraction: list[float] = []
    refit_support_fraction: list[float] = []
    fallback_support_fraction: list[float] = []
    success_support_fraction: list[float] = []

    for layer_id in layer_ids:
        layer_keys = np.asarray(keys[layer_id], dtype=np.float64)
        if layer_keys.shape[0] != n_heads or layer_keys.shape[1] < total_tokens:
            raise ValueError(f"captured keys for layer {layer_id} have the wrong shape")
        for start in gated:
            C = boxes[start]
            forgotten_here = [position for position in forget if position < start]
            forgotten_set = set(forgotten_here)
            kept = [position for position in range(start) if position not in forgotten_set]
            exact, refit, decayed = (
                np.zeros((n_heads, start), dtype=np.float64) for _ in range(3)
            )
            for head in range(n_heads):
                X = layer_keys[head, :start]
                width = (
                    frozen_kpars[int(layer_id)]
                    if frozen_kpars is not None
                    else median_kernel_width(X)
                )
                model = FastOneClassSVM(C=C, ktype="r", kpar=width).seed_from_qp(X)
                full = np.asarray(model.alpha, dtype=np.float64).copy()
                if forgotten_here:
                    n_solves += 1
                    refit_model = FastOneClassSVM(
                        C=C,
                        ktype="r",
                        kpar=width,
                    ).seed_from_qp(X[kept])
                    support_fraction = (len(model.S) + len(model.E)) / start
                    margin_fraction = len(model.S) / start
                    retained_denominator = max(len(kept), 1)
                    affected_support_fraction.append(support_fraction)
                    affected_margin_fraction.append(margin_fraction)
                    refit_support_fraction.append(
                        (len(refit_model.S) + len(refit_model.E))
                        / retained_denominator
                    )
                    refit[head, kept] = np.asarray(
                        refit_model.alpha, dtype=np.float64
                    )
                    if skip_incremental:
                        exact[head] = refit[head]
                        decayed[head] = full
                        decayed[head, forgotten_here] *= float(decay)
                        completed += 1
                        if progress is not None:
                            progress(completed, expected)
                        continue
                    try:
                        model.remove_points(forgotten_here)
                        exact[head, kept] = np.asarray(model.alpha, dtype=np.float64)
                        exact[head, forgotten_here] = 0.0
                        probes = np.vstack([X[kept], X[forgotten_here]])
                        functional_deviation = float(
                            np.max(
                                np.abs(
                                    model.decision_function(probes)
                                    - refit_model.decision_function(probes)
                                )
                            )
                        )
                        max_candidate_deviation = max(
                            max_candidate_deviation,
                            functional_deviation,
                        )
                        if functional_deviation > FUNCTIONAL_TOLERANCE:
                            raise RuntimeError(
                                "post-verification failed: "
                                f"decision deviation {functional_deviation:.3e}"
                            )
                        max_functional_deviation = max(
                            max_functional_deviation,
                            functional_deviation,
                        )
                        success_support_fraction.append(support_fraction)
                    except RuntimeError as exc:
                        n_fallback += 1
                        fallback_support_fraction.append(support_fraction)
                        exact[head] = refit[head]
                        fallback_details.append(
                            {
                                "layer_id": int(layer_id),
                                "start": start,
                                "head": head,
                                "reason": str(exc),
                            }
                        )
                    decayed[head] = full
                    decayed[head, forgotten_here] *= float(decay)
                else:
                    exact[head] = refit[head] = decayed[head] = full
                completed += 1
                if progress is not None:
                    progress(completed, expected)
            cases["exact"][int(layer_id)][start] = exact
            cases["refit"][int(layer_id)][start] = refit
            cases["decay"][int(layer_id)][start] = decayed

    cases["n_solves"] = n_solves
    cases["n_fallback"] = n_fallback
    cases["n_head_gates"] = expected
    cases["box_C"] = (
        float(box_spec) if not isinstance(box_spec, dict) else None
    )
    cases["box_C_by_boundary"] = (
        {str(start): value for start, value in boxes.items()}
        if isinstance(box_spec, dict)
        else None
    )
    cases["objective"] = {
        "box_C": cases["box_C"],
        "box_C_by_boundary": cases["box_C_by_boundary"],
        "box_source": (
            "frozen_decode_session"
            if box_C is not None or box_C_by_boundary is not None
            else (
                "nu_and_prefix_tokens"
                if per_boundary_box
                else "nu_and_total_tokens"
            )
        ),
        "bandwidth_source": (
            "frozen_decode_session"
            if frozen_kpars is not None
            else "per_head_boundary_median"
        ),
        "kpar_by_layer": frozen_kpars,
    }
    cases["feasibility"] = feasibility
    cases["fixed_c_feasible"] = True
    cases["fallback_details"] = fallback_details
    cases["used_refit_fallback"] = bool(n_fallback)
    cases["max_functional_deviation"] = max_functional_deviation
    cases["max_candidate_deviation"] = max_candidate_deviation
    cases["functional_tolerance"] = FUNCTIONAL_TOLERANCE
    def summarize(values):
        return (
            None
            if not values
            else {
                "n": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        )

    cases["partition_diagnostics"] = {
        "affected_support_fraction": summarize(affected_support_fraction),
        "affected_margin_fraction": summarize(affected_margin_fraction),
        "refit_support_fraction": summarize(refit_support_fraction),
        "fallback_support_fraction": summarize(fallback_support_fraction),
        "successful_support_fraction": summarize(success_support_fraction),
    }
    return cases

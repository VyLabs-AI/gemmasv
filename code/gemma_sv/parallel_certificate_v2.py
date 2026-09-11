"""Deterministic CPU-parallel construction of GemmaSV certificate overrides.

This module is intentionally additive.  It does not modify the historical
hash-bound v1 certificate implementation.  The public
``parallel_certificate_overrides`` function preserves the value/schema contract
of :func:`gemma_sv.demo_server.certificate.certificate_overrides` while
partitioning its independent work by ``(layer, head)``.

The process backend uses ``spawn`` and temporary read-only NumPy memory maps.
Workers receive no model objects and do not import Torch, MPS, or MLX.  BLAS
thread-count environment variables are fixed to one while children are
spawned, preventing process-by-thread oversubscription.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from gemma_sv.demo_server import certificate as _certificate


MAX_WORKERS = 32
SUPPORTED_BACKENDS = frozenset({"serial", "process"})
_BLAS_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class _HeadTask:
    layer_id: int
    head: int
    key_path: str
    total_tokens: int
    gated: tuple[int, ...]
    boxes: tuple[tuple[int, float], ...]
    forget: tuple[int, ...]
    decay: float
    skip_incremental: bool
    frozen_kpar: float | None


@dataclass
class _BoundaryHeadResult:
    start: int
    exact: np.ndarray
    refit: np.ndarray
    decay: np.ndarray
    solved: bool
    fallback_reason: str | None
    functional_deviation: float
    candidate_deviation: float
    affected_support_fraction: float | None
    affected_margin_fraction: float | None
    refit_support_fraction: float | None
    fallback_support_fraction: float | None
    successful_support_fraction: float | None


@dataclass
class _HeadResult:
    layer_id: int
    head: int
    boundaries: tuple[_BoundaryHeadResult, ...]


def _validated_execution_settings(
    workers: int,
    backend: str,
    progress: Callable[[int, int], None] | None,
) -> tuple[int, str]:
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ValueError("workers must be an integer")
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}]")
    normalized = str(backend).strip().casefold()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(
            "backend must be one of " + ", ".join(sorted(SUPPORTED_BACKENDS))
        )
    if normalized == "serial" and workers != 1:
        raise ValueError("the serial backend requires workers=1")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    return workers, normalized


def _validated_frozen_kpars(
    layer_ids: Sequence[int],
    kpar_by_layer: Mapping[int, float] | None,
) -> dict[int, float] | None:
    if kpar_by_layer is None:
        return None
    missing = sorted(set(int(layer_id) for layer_id in layer_ids) - set(kpar_by_layer))
    extra = sorted(set(kpar_by_layer) - set(int(layer_id) for layer_id in layer_ids))
    if missing or extra:
        raise ValueError(
            "kpar_by_layer coverage differs from layer_ids; "
            f"missing={missing}, extra={extra}"
        )
    frozen: dict[int, float] = {}
    for layer_id in layer_ids:
        value = float(kpar_by_layer[int(layer_id)])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"kpar_by_layer[{int(layer_id)}] must be finite and positive"
            )
        frozen[int(layer_id)] = value
    return frozen


def _validated_keys(
    keys: Mapping[int, np.ndarray],
    layer_ids: Sequence[int],
    *,
    total_tokens: int,
    n_heads: int,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for layer_id in layer_ids:
        resolved = int(layer_id)
        layer_keys = np.asarray(keys[layer_id], dtype=np.float64)
        if (
            layer_keys.ndim != 3
            or layer_keys.shape[0] != n_heads
            or layer_keys.shape[1] < total_tokens
        ):
            raise ValueError(
                f"captured keys for layer {layer_id} have the wrong shape"
            )
        result[resolved] = np.ascontiguousarray(
            layer_keys[:, :total_tokens],
            dtype=np.float64,
        )
    return result


@contextmanager
def _single_thread_blas_environment() -> Iterable[None]:
    previous = {name: os.environ.get(name) for name in _BLAS_THREAD_ENVIRONMENT}
    try:
        for name in _BLAS_THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _temporary_key_store(
    keys: Mapping[int, np.ndarray],
    layer_ids: Sequence[int],
) -> Iterable[dict[int, str]]:
    with tempfile.TemporaryDirectory(prefix="gemmasv-cert-v2-") as directory:
        root = Path(directory)
        root.chmod(0o700)
        paths: dict[int, str] = {}
        for index, layer_id in enumerate(layer_ids):
            resolved = int(layer_id)
            path = root / f"{index:04d}-{resolved}.npy"
            with path.open("wb") as handle:
                np.save(handle, keys[resolved], allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o600)
            paths[resolved] = str(path)
        yield paths


def _solve_head(task: _HeadTask) -> _HeadResult:
    # Imports remain local so spawned workers load only the CPU certificate path.
    from cp_svm import FastOneClassSVM

    layer_keys = np.load(task.key_path, mmap_mode="r", allow_pickle=False)
    boxes = dict(task.boxes)
    boundaries: list[_BoundaryHeadResult] = []
    try:
        for start in task.gated:
            C = boxes[start]
            forgotten_here = [
                position for position in task.forget if position < start
            ]
            forgotten_set = set(forgotten_here)
            kept = [
                position
                for position in range(start)
                if position not in forgotten_set
            ]
            exact = np.zeros(start, dtype=np.float64)
            refit = np.zeros(start, dtype=np.float64)
            decayed = np.zeros(start, dtype=np.float64)
            X = np.asarray(layer_keys[task.head, :start], dtype=np.float64)
            width = (
                task.frozen_kpar
                if task.frozen_kpar is not None
                else _certificate.median_kernel_width(X)
            )
            model = FastOneClassSVM(
                C=C,
                ktype="r",
                kpar=width,
            ).seed_from_qp(X)
            full = np.asarray(model.alpha, dtype=np.float64).copy()

            solved = bool(forgotten_here)
            fallback_reason: str | None = None
            functional_deviation = 0.0
            candidate_deviation = 0.0
            affected_support_fraction: float | None = None
            affected_margin_fraction: float | None = None
            refit_support_fraction: float | None = None
            fallback_support_fraction: float | None = None
            successful_support_fraction: float | None = None

            if forgotten_here:
                refit_model = FastOneClassSVM(
                    C=C,
                    ktype="r",
                    kpar=width,
                ).seed_from_qp(X[kept])
                affected_support_fraction = (
                    len(model.S) + len(model.E)
                ) / start
                affected_margin_fraction = len(model.S) / start
                retained_denominator = max(len(kept), 1)
                refit_support_fraction = (
                    len(refit_model.S) + len(refit_model.E)
                ) / retained_denominator
                refit[kept] = np.asarray(
                    refit_model.alpha,
                    dtype=np.float64,
                )
                if task.skip_incremental:
                    exact[:] = refit
                else:
                    try:
                        model.remove_points(forgotten_here)
                        exact[kept] = np.asarray(
                            model.alpha,
                            dtype=np.float64,
                        )
                        exact[forgotten_here] = 0.0
                        probes = np.vstack([X[kept], X[forgotten_here]])
                        candidate_deviation = float(
                            np.max(
                                np.abs(
                                    model.decision_function(probes)
                                    - refit_model.decision_function(probes)
                                )
                            )
                        )
                        if (
                            candidate_deviation
                            > _certificate.FUNCTIONAL_TOLERANCE
                        ):
                            raise RuntimeError(
                                "post-verification failed: "
                                f"decision deviation {candidate_deviation:.3e}"
                            )
                        functional_deviation = candidate_deviation
                        successful_support_fraction = (
                            affected_support_fraction
                        )
                    except RuntimeError as exc:
                        fallback_reason = str(exc)
                        fallback_support_fraction = affected_support_fraction
                        exact[:] = refit
                decayed[:] = full
                decayed[forgotten_here] *= float(task.decay)
            else:
                exact[:] = full
                refit[:] = full
                decayed[:] = full

            boundaries.append(
                _BoundaryHeadResult(
                    start=start,
                    exact=exact,
                    refit=refit,
                    decay=decayed,
                    solved=solved,
                    fallback_reason=fallback_reason,
                    functional_deviation=functional_deviation,
                    candidate_deviation=candidate_deviation,
                    affected_support_fraction=affected_support_fraction,
                    affected_margin_fraction=affected_margin_fraction,
                    refit_support_fraction=refit_support_fraction,
                    fallback_support_fraction=fallback_support_fraction,
                    successful_support_fraction=successful_support_fraction,
                )
            )
    finally:
        del layer_keys
    return _HeadResult(
        layer_id=task.layer_id,
        head=task.head,
        boundaries=tuple(boundaries),
    )


def _tasks(
    paths: Mapping[int, str],
    layer_ids: Sequence[int],
    *,
    total_tokens: int,
    n_heads: int,
    gated: Sequence[int],
    boxes: Mapping[int, float],
    forget: tuple[int, ...],
    decay: float,
    skip_incremental: bool,
    frozen_kpars: Mapping[int, float] | None,
) -> list[_HeadTask]:
    canonical_boxes = tuple(
        (int(start), float(boxes[start])) for start in gated
    )
    return [
        _HeadTask(
            layer_id=int(layer_id),
            head=head,
            key_path=paths[int(layer_id)],
            total_tokens=total_tokens,
            gated=tuple(int(start) for start in gated),
            boxes=canonical_boxes,
            forget=forget,
            decay=float(decay),
            skip_incremental=bool(skip_incremental),
            frozen_kpar=(
                None
                if frozen_kpars is None
                else float(frozen_kpars[int(layer_id)])
            ),
        )
        for layer_id in layer_ids
        for head in range(n_heads)
    ]


def _run_process_tasks(
    tasks: Sequence[_HeadTask],
    *,
    workers: int,
    progress: Callable[[int, int], None] | None,
    expected: int,
) -> list[_HeadResult]:
    context = multiprocessing.get_context("spawn")
    results: list[_HeadResult] = []
    completed = 0
    with _single_thread_blas_environment():
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(_solve_head, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += len(result.boundaries)
                if progress is not None:
                    progress(completed, expected)
    return results


def _summary(values: Sequence[float]) -> dict[str, float | int] | None:
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


def _merge_head_results(
    results: Sequence[_HeadResult],
    *,
    layer_ids: Sequence[int],
    gated: Sequence[int],
    n_heads: int,
    box_spec: float | dict[int, float],
    boxes: Mapping[int, float],
    feasibility: Sequence[Mapping[str, Any]],
    box_C: float | None,
    box_C_by_boundary: Mapping[int, float] | None,
    per_boundary_box: bool,
    frozen_kpars: Mapping[int, float] | None,
) -> dict[str, Any]:
    by_identity = {
        (int(result.layer_id), int(result.head)): {
            boundary.start: boundary for boundary in result.boundaries
        }
        for result in results
    }
    expected_identities = {
        (int(layer_id), head)
        for layer_id in layer_ids
        for head in range(n_heads)
    }
    if set(by_identity) != expected_identities:
        raise RuntimeError("parallel certificate head coverage differs")

    cases: dict[str, Any] = {
        name: {int(layer_id): {} for layer_id in layer_ids}
        for name in ("exact", "refit", "decay")
    }
    n_solves = n_fallback = 0
    max_functional_deviation = 0.0
    max_candidate_deviation = 0.0
    fallback_details: list[dict[str, int | str | float]] = []
    affected_support_fraction: list[float] = []
    affected_margin_fraction: list[float] = []
    refit_support_fraction: list[float] = []
    fallback_support_fraction: list[float] = []
    success_support_fraction: list[float] = []
    expected = len(layer_ids) * len(gated) * n_heads

    for layer_id in layer_ids:
        resolved_layer = int(layer_id)
        for start in gated:
            exact = np.zeros((n_heads, start), dtype=np.float64)
            refit = np.zeros((n_heads, start), dtype=np.float64)
            decayed = np.zeros((n_heads, start), dtype=np.float64)
            for head in range(n_heads):
                boundary = by_identity[(resolved_layer, head)].get(start)
                if boundary is None:
                    raise RuntimeError(
                        "parallel certificate boundary coverage differs"
                    )
                exact[head] = boundary.exact
                refit[head] = boundary.refit
                decayed[head] = boundary.decay
                n_solves += int(boundary.solved)
                max_functional_deviation = max(
                    max_functional_deviation,
                    boundary.functional_deviation,
                )
                max_candidate_deviation = max(
                    max_candidate_deviation,
                    boundary.candidate_deviation,
                )
                if boundary.affected_support_fraction is not None:
                    affected_support_fraction.append(
                        boundary.affected_support_fraction
                    )
                if boundary.affected_margin_fraction is not None:
                    affected_margin_fraction.append(
                        boundary.affected_margin_fraction
                    )
                if boundary.refit_support_fraction is not None:
                    refit_support_fraction.append(
                        boundary.refit_support_fraction
                    )
                if boundary.fallback_support_fraction is not None:
                    fallback_support_fraction.append(
                        boundary.fallback_support_fraction
                    )
                if boundary.successful_support_fraction is not None:
                    success_support_fraction.append(
                        boundary.successful_support_fraction
                    )
                if boundary.fallback_reason is not None:
                    n_fallback += 1
                    fallback_details.append(
                        {
                            "layer_id": resolved_layer,
                            "start": start,
                            "head": head,
                            "reason": boundary.fallback_reason,
                        }
                    )
            cases["exact"][resolved_layer][start] = exact
            cases["refit"][resolved_layer][start] = refit
            cases["decay"][resolved_layer][start] = decayed

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
        "kpar_by_layer": (
            None if frozen_kpars is None else dict(frozen_kpars)
        ),
    }
    cases["feasibility"] = list(feasibility)
    cases["fixed_c_feasible"] = True
    cases["fallback_details"] = fallback_details
    cases["used_refit_fallback"] = bool(n_fallback)
    cases["max_functional_deviation"] = max_functional_deviation
    cases["max_candidate_deviation"] = max_candidate_deviation
    cases["functional_tolerance"] = _certificate.FUNCTIONAL_TOLERANCE
    cases["partition_diagnostics"] = {
        "affected_support_fraction": _summary(affected_support_fraction),
        "affected_margin_fraction": _summary(affected_margin_fraction),
        "refit_support_fraction": _summary(refit_support_fraction),
        "fallback_support_fraction": _summary(fallback_support_fraction),
        "successful_support_fraction": _summary(success_support_fraction),
    }
    return cases


def parallel_certificate_overrides(
    keys: Mapping[int, np.ndarray],
    layer_ids: Sequence[int],
    forget_positions: Sequence[int],
    total_tokens: int,
    n_heads: int,
    *,
    nu: float = 0.3,
    chunk: int = 128,
    decay: float = _certificate.DECAY,
    progress: Callable[[int, int], None] | None = None,
    skip_incremental: bool = False,
    box_C: float | None = None,
    box_C_by_boundary: Mapping[int, float] | None = None,
    per_boundary_box: bool = False,
    kpar_by_layer: Mapping[int, float] | None = None,
    workers: int = 1,
    backend: str = "process",
) -> dict[str, Any]:
    """Return certificate overrides with the historical v1 value/schema.

    ``workers=1`` executes the same per-head kernel in-process and is the serial
    oracle for this additive implementation.  ``backend="process"`` with more
    than one worker uses spawned CPU-only children and read-only memory-mapped
    key arrays.  The process completion order cannot affect the merged value.
    """

    workers, backend = _validated_execution_settings(
        workers,
        backend,
        progress,
    )
    if total_tokens <= 0 or n_heads <= 0:
        raise ValueError("certificate dimensions must be positive")
    if not layer_ids:
        raise ValueError("at least one global layer is required")

    box_spec, feasibility = _certificate.fixed_c_feasibility(
        total_tokens,
        forget_positions,
        nu=nu,
        chunk=chunk,
        box_C=box_C,
        box_C_by_boundary=box_C_by_boundary,
        per_boundary_box=per_boundary_box,
    )
    if any(not item["feasible"] for item in feasibility):
        raise _certificate.CertificateFeasibilityError(feasibility)
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
    frozen_kpars = _validated_frozen_kpars(layer_ids, kpar_by_layer)
    validated_keys = _validated_keys(
        keys,
        layer_ids,
        total_tokens=total_tokens,
        n_heads=n_heads,
    )

    # No worker or scratch resource is needed if the certificate has no gates.
    if not gated:
        results = [
            _HeadResult(int(layer_id), head, ())
            for layer_id in layer_ids
            for head in range(n_heads)
        ]
    elif workers == 1:
        synthetic_paths = {
            int(layer_id): "" for layer_id in layer_ids
        }
        with _temporary_key_store(validated_keys, layer_ids) as paths:
            tasks = _tasks(
                {**synthetic_paths, **paths},
                layer_ids,
                total_tokens=total_tokens,
                n_heads=n_heads,
                gated=gated,
                boxes=boxes,
                forget=tuple(
                    sorted({int(position) for position in forget_positions})
                ),
                decay=decay,
                skip_incremental=skip_incremental,
                frozen_kpars=frozen_kpars,
            )
            results = []
            completed = 0
            expected = len(tasks) * len(gated)
            for task in tasks:
                result = _solve_head(task)
                results.append(result)
                completed += len(result.boundaries)
                if progress is not None:
                    progress(completed, expected)
    else:
        if backend != "process":
            raise ValueError("parallel execution requires backend='process'")
        with _temporary_key_store(validated_keys, layer_ids) as paths:
            tasks = _tasks(
                paths,
                layer_ids,
                total_tokens=total_tokens,
                n_heads=n_heads,
                gated=gated,
                boxes=boxes,
                forget=tuple(
                    sorted({int(position) for position in forget_positions})
                ),
                decay=decay,
                skip_incremental=skip_incremental,
                frozen_kpars=frozen_kpars,
            )
            results = _run_process_tasks(
                tasks,
                workers=workers,
                progress=progress,
                expected=len(tasks) * len(gated),
            )

    return _merge_head_results(
        results,
        layer_ids=layer_ids,
        gated=gated,
        n_heads=n_heads,
        box_spec=box_spec,
        boxes=boxes,
        feasibility=feasibility,
        box_C=box_C,
        box_C_by_boundary=box_C_by_boundary,
        per_boundary_box=per_boundary_box,
        frozen_kpars=frozen_kpars,
    )


__all__ = [
    "MAX_WORKERS",
    "SUPPORTED_BACKENDS",
    "parallel_certificate_overrides",
]

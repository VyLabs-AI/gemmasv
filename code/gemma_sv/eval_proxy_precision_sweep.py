"""Calibrate one residual-gated FP32 proxy policy, then evaluate it once.

The calibration split sweeps a predeclared FISTA/partition grid.  A policy may
use the proxy only when solver residuals pass its predeclared cutoff; otherwise
it falls back to the matched float64 fixed-C refit.  The locked validation split
is run only for the selected policy.  This is a residual-qualified
approximation study, not a certificate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_persistent_deletion_baselines import (
    QueryState,
    _build_probes,
    _score_probe_suite,
    _seed_everything,
    _synchronize,
)
from gemma_sv.persistent_deletion import (
    build_synthetic_record_context,
    full_vocabulary_kl,
    method_storage_report,
)
from gemma_sv.recovery_protocol import MODEL_REVISION


SCHEMA_VERSION = 1
CALIBRATION_INDICES = (0, 2, 4, 6)
VALIDATION_INDICES = (1, 3, 5, 7)
FISTA_ITERATIONS = (20, 40, 80, 160, 320, 640, 1280)
PARTITION_CUTOFFS = (1e-4, 3e-4, 1e-3, 3e-3)
RESIDUAL_CUTOFFS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
MAX_REFERENCE_KL_NATS = 1e-3
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / "whole_record_synthetic_v1.json"
)
DEFAULT_OUTPUT = Path("outputs/gemma_sv_proxy_precision/frontier.json")


def _project_capped_simplex(
    values: Sequence[float],
    box_C: float,
) -> np.ndarray:
    """Euclidean projection onto ``sum(alpha)=1, 0<=alpha<=C``."""

    z = np.asarray(values, dtype=np.float64)
    C = float(box_C)
    if z.ndim != 1 or not len(z):
        raise ValueError("projection requires one non-empty vector")
    if not math.isfinite(C) or C <= 0 or len(z) * C < 1.0 - 1e-12:
        raise ValueError("capped-simplex constraints are infeasible")
    lower = float(np.min(z) - C - 1.0)
    upper = float(np.max(z) + 1.0)
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        mass = float(np.clip(z - midpoint, 0.0, C).sum())
        if mass > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(z - 0.5 * (lower + upper), 0.0, C)
    # Remove the last few ulps of bisection error without changing the active set.
    error = 1.0 - float(projected.sum())
    free = np.flatnonzero((projected > 1e-12) & (projected < C - 1e-12))
    if len(free):
        projected[free] += error / len(free)
    return projected


def _linear_minimizer(gradient: np.ndarray, box_C: float) -> np.ndarray:
    """Solve the capped-simplex linear oracle used by the FW duality gap."""

    result = np.zeros_like(gradient, dtype=np.float64)
    remaining = 1.0
    for index in np.argsort(gradient, kind="stable"):
        assigned = min(float(box_C), remaining)
        result[int(index)] = assigned
        remaining -= assigned
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        raise ValueError("capped-simplex linear oracle is infeasible")
    return result


def svdd_residual_diagnostics(
    gram: Sequence[Sequence[float]],
    alpha: Sequence[float],
    box_C: float,
    *,
    partition_cutoff: float,
) -> dict[str, Any]:
    """Return feasibility, KKT, projected-gradient, and FW-gap diagnostics."""

    K = np.asarray(gram, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    C = float(box_C)
    cutoff = float(partition_cutoff)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != a.shape[0]:
        raise ValueError("gram and alpha shapes do not agree")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(a)):
        raise ValueError("solver diagnostics require finite inputs")
    if not 0 < cutoff < C:
        raise ValueError("partition_cutoff must be in (0, C)")

    equality = abs(float(a.sum()) - 1.0)
    lower_violation = max(0.0, float(-np.min(a)))
    upper_violation = max(0.0, float(np.max(a) - C))
    gradient = 2.0 * (K @ a) - np.diag(K)
    margin = (a > cutoff) & (a < C - cutoff)
    lower = a <= cutoff
    upper = a >= C - cutoff

    if np.any(margin):
        rho = -float(np.mean(gradient[margin]))
    else:
        rho_lower = (
            float(np.max(-gradient[lower])) if np.any(lower) else -math.inf
        )
        rho_upper = (
            float(np.min(-gradient[upper])) if np.any(upper) else math.inf
        )
        if math.isfinite(rho_lower) and math.isfinite(rho_upper):
            rho = 0.5 * (rho_lower + rho_upper)
        elif math.isfinite(rho_lower):
            rho = rho_lower
        elif math.isfinite(rho_upper):
            rho = rho_upper
        else:
            rho = -float(np.mean(gradient))
    stationarity = gradient + rho
    kkt_terms = [0.0]
    if np.any(margin):
        kkt_terms.append(float(np.max(np.abs(stationarity[margin]))))
    if np.any(lower):
        kkt_terms.append(float(np.max(np.maximum(0.0, -stationarity[lower]))))
    if np.any(upper):
        kkt_terms.append(float(np.max(np.maximum(0.0, stationarity[upper]))))
    kkt = max(kkt_terms)

    projected = _project_capped_simplex(a - gradient, C)
    projected_gradient = float(np.max(np.abs(a - projected)))
    linear_solution = _linear_minimizer(gradient, C)
    duality_gap = max(0.0, float(gradient @ (a - linear_solution)))
    objective = float(a @ K @ a - np.diag(K) @ a)
    return {
        "feasible": bool(
            equality <= 1e-5
            and lower_violation <= 1e-7
            and upper_violation <= 1e-7
        ),
        "equality_residual": equality,
        "lower_box_violation": lower_violation,
        "upper_box_violation": upper_violation,
        "kkt_residual": kkt,
        "projected_gradient_inf": projected_gradient,
        "duality_gap": duality_gap,
        "objective": objective,
        "rho": rho,
        "partition": {
            "margin": int(np.count_nonzero(margin)),
            "error": int(np.count_nonzero(upper)),
            "reserve": int(np.count_nonzero(lower)),
            "cutoff": cutoff,
        },
    }


def residual_qualifies(
    diagnostics: Mapping[str, Any],
    residual_cutoff: float,
) -> bool:
    """Apply one predeclared, scale-consistent residual gate."""

    cutoff = float(residual_cutoff)
    if not diagnostics.get("feasible", False):
        return False
    if int(diagnostics.get("invalid_gate_rows", 0)) > 0:
        return False
    return (
        float(diagnostics["max_equality_residual"]) <= min(1e-5, cutoff)
        and float(diagnostics["max_box_violation"]) <= min(1e-7, cutoff)
        and float(diagnostics["max_kkt_residual"]) <= cutoff
        and float(diagnostics["max_projected_gradient_inf"]) <= cutoff
        and float(diagnostics["max_duality_gap"]) <= cutoff**2
    )


def _hybrid_row(
    observation: Mapping[str, Any],
    residual_cutoff: float,
) -> dict[str, Any]:
    qualified = residual_qualifies(
        observation["diagnostics"],
        residual_cutoff,
    )
    metrics = observation["metrics"]
    timing = observation["timing"]
    if qualified:
        kl = float(metrics["max_kl_refit64_to_proxy_nats"])
        retained_drift = float(metrics["retained_drift_from_refit64_nats"])
        update = float(timing["proxy_update_seconds"])
        query = float(timing["proxy_query_seconds"])
    else:
        kl = 0.0
        retained_drift = 0.0
        # Residuals are known only after the proxy solve, so a fallback pays both.
        update = float(timing["proxy_update_seconds"]) + float(
            timing["shared_exact_refit_update_seconds"]
        )
        query = float(timing["refit_query_seconds"])
    return {
        "adapter_index": int(observation["adapter_index"]),
        "record_index": int(observation["record_index"]),
        "used_proxy": qualified,
        "used_exact_refit_fallback": not qualified,
        "max_kl_refit64_to_hybrid_nats": kl,
        "retained_drift_from_refit64_nats": retained_drift,
        "update_seconds": update,
        "query_seconds": query,
        "end_to_end_seconds": update + query,
        "violates_reference_kl": kl > MAX_REFERENCE_KL_NATS,
    }


def select_calibration_policy(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the fastest predeclared zero-violation calibration policy."""

    if not observations:
        raise ValueError("calibration observations are empty")
    expected_contexts = {
        (int(row["adapter_index"]), int(row["record_index"]))
        for row in observations
    }
    solver_configs = sorted(
        {
            (
                int(row["solver"]["fista_iterations"]),
                float(row["solver"]["partition_cutoff"]),
            )
            for row in observations
        }
    )
    frontier = []
    for iterations, partition_cutoff in solver_configs:
        selected = [
            row
            for row in observations
            if int(row["solver"]["fista_iterations"]) == iterations
            and float(row["solver"]["partition_cutoff"]) == partition_cutoff
        ]
        observed_contexts = {
            (int(row["adapter_index"]), int(row["record_index"]))
            for row in selected
        }
        if observed_contexts != expected_contexts:
            raise ValueError("a calibration solver configuration is incomplete")
        for residual_cutoff in RESIDUAL_CUTOFFS:
            rows = [_hybrid_row(row, residual_cutoff) for row in selected]
            fallbacks = sum(row["used_exact_refit_fallback"] for row in rows)
            violations = sum(row["violates_reference_kl"] for row in rows)
            frontier.append(
                {
                    "policy": {
                        "fista_iterations": iterations,
                        "partition_cutoff": partition_cutoff,
                        "residual_cutoff": residual_cutoff,
                        "max_reference_kl_nats": MAX_REFERENCE_KL_NATS,
                    },
                    "contexts": len(rows),
                    "context_violations": violations,
                    "fallbacks": fallbacks,
                    "fallback_rate": fallbacks / len(rows),
                    "max_kl_refit64_to_hybrid_nats": max(
                        row["max_kl_refit64_to_hybrid_nats"] for row in rows
                    ),
                    "mean_retained_abs_drift_nats": statistics.fmean(
                        abs(row["retained_drift_from_refit64_nats"])
                        for row in rows
                    ),
                    "mean_update_seconds": statistics.fmean(
                        row["update_seconds"] for row in rows
                    ),
                    "mean_query_seconds": statistics.fmean(
                        row["query_seconds"] for row in rows
                    ),
                    "mean_end_to_end_seconds": statistics.fmean(
                        row["end_to_end_seconds"] for row in rows
                    ),
                }
            )
    passing = [
        row
        for row in frontier
        if row["context_violations"] == 0
        and row["fallbacks"] < row["contexts"]
    ]
    passing.sort(
        key=lambda row: (
            row["mean_end_to_end_seconds"],
            row["fallback_rate"],
            row["policy"]["fista_iterations"],
            row["policy"]["partition_cutoff"],
            row["policy"]["residual_cutoff"],
        )
    )
    return {
        "selection_rule": (
            "fastest nontrivial calibration policy (at least one qualified "
            "proxy context) with zero context violations of "
            "max KL(refit64 || hybrid) <= 1e-3"
        ),
        "selected_policy": passing[0]["policy"] if passing else None,
        "status": "selected" if passing else "negative_frontier",
        "frontier": frontier,
    }


def evaluate_locked_policy(
    observations: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize the one policy evaluated on the locked validation records."""

    rows = [_hybrid_row(row, float(policy["residual_cutoff"])) for row in observations]
    fallbacks = sum(row["used_exact_refit_fallback"] for row in rows)
    violations = sum(row["violates_reference_kl"] for row in rows)
    return {
        "policy": dict(policy),
        "contexts": len(rows),
        "context_violations": violations,
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / len(rows) if rows else 0.0,
        "max_kl_refit64_to_hybrid_nats": (
            max(row["max_kl_refit64_to_hybrid_nats"] for row in rows)
            if rows
            else None
        ),
        "mean_retained_abs_drift_nats": (
            statistics.fmean(
                abs(row["retained_drift_from_refit64_nats"]) for row in rows
            )
            if rows
            else None
        ),
        "mean_end_to_end_seconds": (
            statistics.fmean(row["end_to_end_seconds"] for row in rows)
            if rows
            else None
        ),
        "rows": rows,
    }


def _rbf_gram(keys: np.ndarray, kpar: float) -> np.ndarray:
    differences = keys[:, None, :] - keys[None, :, :]
    squared = np.sum(differences * differences, axis=-1)
    return np.exp(-squared / float(kpar) ** 2)


def _state_diagnostics(
    source_memory: Any,
    proxy_memory: Any,
    forget_positions: Sequence[int],
    *,
    partition_cutoff: float,
) -> dict[str, Any]:
    rows = []
    invalid = 0
    forgotten = set(int(position) for position in forget_positions)
    for layer_id, source_session in source_memory.layer_sessions.items():
        proxy_session = proxy_memory.layer_sessions[layer_id]
        keys = source_session.kf.detach().cpu().double().numpy()
        for boundary, (alphas, valid_rows) in sorted(proxy_session.gates.items()):
            alpha_np = alphas.detach().cpu().double().numpy()
            valid_np = valid_rows.detach().cpu().numpy().astype(bool)
            retained = [
                position for position in range(int(boundary))
                if position not in forgotten
            ]
            for head in range(alpha_np.shape[0]):
                if not valid_np[head]:
                    invalid += 1
                    continue
                gram = _rbf_gram(
                    keys[head, retained],
                    float(source_session.kpar),
                )
                diagnostics = svdd_residual_diagnostics(
                    gram,
                    alpha_np[head, retained],
                    float(source_session.box_C),
                    partition_cutoff=partition_cutoff,
                )
                rows.append(
                    {
                        "layer_id": int(layer_id),
                        "boundary": int(boundary),
                        "head": int(head),
                        **diagnostics,
                    }
                )
    if not rows:
        failed_residual = 1e300
        return {
            "feasible": False,
            "invalid_gate_rows": invalid,
            "diagnosed_gate_rows": 0,
            "max_equality_residual": failed_residual,
            "max_box_violation": failed_residual,
            "max_kkt_residual": failed_residual,
            "max_projected_gradient_inf": failed_residual,
            "max_duality_gap": failed_residual,
            "gate_rows": [],
        }
    return {
        "feasible": all(row["feasible"] for row in rows),
        "invalid_gate_rows": invalid,
        "diagnosed_gate_rows": len(rows),
        "max_equality_residual": max(row["equality_residual"] for row in rows),
        "max_box_violation": max(
            max(row["lower_box_violation"], row["upper_box_violation"])
            for row in rows
        ),
        "max_kkt_residual": max(row["kkt_residual"] for row in rows),
        "max_projected_gradient_inf": max(
            row["projected_gradient_inf"] for row in rows
        ),
        "max_duality_gap": max(row["duality_gap"] for row in rows),
        "gate_rows": rows,
    }


def _score_metrics(probes, reference_scores, proxy_scores) -> dict[str, float]:
    kls = []
    retained_drift = None
    for probe in probes:
        reference = reference_scores[probe.probe_id]
        proxy = proxy_scores[probe.probe_id]
        kls.append(
            full_vocabulary_kl(
                reference["first_log_probs"],
                proxy["first_log_probs"],
            )
        )
        if probe.kind == "retained":
            retained_drift = float(
                proxy["mean_log_probability"]
                - reference["mean_log_probability"]
            )
    if retained_drift is None:
        raise RuntimeError("proxy sweep requires one retained probe")
    return {
        "max_kl_refit64_to_proxy_nats": max(kls),
        "mean_kl_refit64_to_proxy_nats": statistics.fmean(kls),
        "retained_drift_from_refit64_nats": retained_drift,
    }


def _peak_memory() -> dict[str, int | str]:
    # ru_maxrss is bytes on Darwin and KiB on Linux.
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    scale = 1 if os.uname().sysname == "Darwin" else 1024
    result: dict[str, int | str] = {
        "process_peak_resident_bytes": raw * scale,
        "scope": "process high-water mark; monotone across the run",
    }
    try:
        import torch
    except ImportError:
        return result
    if torch.cuda.is_available():
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "driver_allocated_memory"):
        result["mps_driver_allocated_bytes"] = int(mps.driver_allocated_memory())
    return result


def _evaluate_solver_config(
    runtime: GemmaRuntime,
    original_memory: Any,
    refit_state: Any,
    refit_scores: Mapping[str, Mapping[str, Any]],
    probes,
    forget_positions: Sequence[int],
    *,
    adapter_index: int,
    record_index: int,
    record_id: str,
    fista_iterations: int,
    partition_cutoff: float,
    shared_exact_refit_update_seconds: float,
    refit_query_seconds: float,
) -> dict[str, Any]:
    solver_seed = 10_000 * (int(adapter_index) + 1) + int(record_index)
    _seed_everything(solver_seed)
    for layer in runtime.layers.values():
        layer.self_attn.fista_iters = int(fista_iterations)
        layer.self_attn.partition_tol = float(partition_cutoff)
    _synchronize(runtime.config.device)
    started = time.perf_counter()
    proxy_state = runtime.delete_persistent(
        original_memory,
        tuple(forget_positions),
        kind="residual_qualified_fp32_proxy",
    )
    _synchronize(runtime.config.device)
    update_seconds = time.perf_counter() - started

    _synchronize(runtime.config.device)
    started = time.perf_counter()
    proxy_scores = _score_probe_suite(
        runtime,
        QueryState(proxy_state),
        probes,
    )
    _synchronize(runtime.config.device)
    query_seconds = time.perf_counter() - started
    return {
        "adapter_index": adapter_index,
        "record_index": record_index,
        "record_id": record_id,
        "solver": {
            "fista_iterations": int(fista_iterations),
            "partition_cutoff": float(partition_cutoff),
            "power_iteration_seed": solver_seed,
            "objective": "frozen session kpar and box_C",
        },
        "metrics": _score_metrics(probes, refit_scores, proxy_scores),
        "diagnostics": _state_diagnostics(
            original_memory,
            proxy_state,
            forget_positions,
            partition_cutoff=partition_cutoff,
        ),
        "timing": {
            "clock": "time.perf_counter",
            "synchronized_device": runtime.config.device,
            "proxy_update_seconds": update_seconds,
            "proxy_query_seconds": query_seconds,
            "shared_exact_refit_update_seconds": (
                shared_exact_refit_update_seconds
            ),
            "refit_query_seconds": refit_query_seconds,
            "fallback_cost_scope": (
                "proxy solve plus shared decrement/refit audit; conservative "
                "because a standalone refit is not timed separately"
            ),
        },
        "storage": {
            "proxy": method_storage_report(original_memory, proxy_state),
            "refit64": method_storage_report(original_memory, refit_state),
        },
        "peak_memory": _peak_memory(),
    }


def _run_record(
    runtime: GemmaRuntime,
    manifest_records: Sequence[Mapping[str, Any]],
    record_index: int,
    *,
    adapter_index: int,
    seed: int,
    window: int,
    n_fill: int,
    prefix_fillers: int,
    copies: int,
    solver_configs: Iterable[tuple[int, float]],
) -> list[dict[str, Any]]:
    spec = manifest_records[record_index]
    _seed_everything(seed + adapter_index * 10_000 + record_index)
    context = build_synthetic_record_context(
        runtime.tokenizer,
        spec,
        runtime.fillers,
        window=window,
        min_fillers=n_fill,
        prefix_fillers=prefix_fillers,
        copies=copies,
    )
    probes = _build_probes(runtime, spec)
    original = runtime.prefill_persistent(list(context.original_token_ids))

    _synchronize(runtime.config.device)
    started = time.perf_counter()
    overrides, reference_states = runtime.persistent_certificate_states(
        original,
        context.forget_positions,
    )
    _synchronize(runtime.config.device)
    reference_update = time.perf_counter() - started
    if overrides.get("objective", {}).get("bandwidth_source") != (
        "frozen_decode_session"
    ):
        raise RuntimeError("float64 reference did not use the frozen bandwidth")

    refit_state = reference_states["refit"]
    _synchronize(runtime.config.device)
    started = time.perf_counter()
    refit_scores = _score_probe_suite(runtime, QueryState(refit_state), probes)
    _synchronize(runtime.config.device)
    refit_query = time.perf_counter() - started

    rows = []
    for fista_iterations, partition_cutoff in solver_configs:
        rows.append(
            _evaluate_solver_config(
                runtime,
                original,
                refit_state,
                refit_scores,
                probes,
                context.forget_positions,
                adapter_index=adapter_index,
                record_index=record_index,
                record_id=str(spec["record_id"]),
                fista_iterations=fista_iterations,
                partition_cutoff=partition_cutoff,
                shared_exact_refit_update_seconds=reference_update,
                refit_query_seconds=refit_query,
            )
        )
    return rows


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _comma_values(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one comma-separated value")
    return result


def _run_matrix(adapters: Sequence[str], devices: Sequence[str]):
    if len(adapters) == len(devices):
        return list(zip(adapters, devices))
    if len(devices) == 1:
        return [(adapter, devices[0]) for adapter in adapters]
    raise ValueError("--devices must contain one value or match --adapters")


def _release_runtime(runtime: GemmaRuntime | None) -> None:
    if runtime is not None:
        runtime.model = None
        runtime.tokenizer = None
        runtime.layers = {}
        runtime.controller = None
        runtime.loaded = False
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _run_group(
    matrix: Sequence[tuple[str, str]],
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    solver_configs: Sequence[tuple[int, float]],
    *,
    args,
    report: dict[str, Any],
    section: str,
    output: Path,
) -> list[dict[str, Any]]:
    observations = []
    for adapter_index, (adapter, device) in enumerate(matrix):
        runtime = None
        try:
            runtime = GemmaRuntime(
                RuntimeConfig(
                    model_id=args.model,
                    model_revision=args.model_revision,
                    lora_path=adapter,
                    device=device,
                    dtype="float64",
                    generation_tokens=1,
                    window=args.window,
                    copies=args.copies,
                )
            )
            runtime.ensure_loaded()
            for record_index in indices:
                print(
                    f"[{section}] adapter {adapter_index} record {record_index}",
                    flush=True,
                )
                rows = _run_record(
                    runtime,
                    records,
                    record_index,
                    adapter_index=adapter_index,
                    seed=args.seed,
                    window=args.window,
                    n_fill=args.n_fill,
                    prefix_fillers=args.prefix_fillers,
                    copies=args.copies,
                    solver_configs=solver_configs,
                )
                observations.extend(rows)
                report[section]["observations"] = observations
                _atomic_write(output, report)
        finally:
            _release_runtime(runtime)
    return observations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--n-fill", type=int, default=22)
    parser.add_argument("--prefix-fillers", type=int, default=8)
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use one adapter, one calibration record, and the smallest solver",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        adapters = _comma_values(args.adapters)
        devices = _comma_values(args.devices)
        matrix = _run_matrix(adapters, devices)
    except ValueError as error:
        parser.error(str(error))
    if not args.smoke and len(matrix) != 3:
        parser.error("the locked paper sweep requires exactly three adapters")
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite to replace it")

    raw_manifest = args.manifest.read_bytes()
    manifest = json.loads(raw_manifest)
    records = list(manifest["records"])
    if len(records) < 8:
        parser.error("the locked split requires at least eight manifest records")
    calibration_indices = (CALIBRATION_INDICES[0],) if args.smoke else CALIBRATION_INDICES
    matrix = matrix[:1] if args.smoke else matrix
    calibration_configs = (
        ((FISTA_ITERATIONS[0], PARTITION_CUTOFFS[0]),)
        if args.smoke
        else tuple(
            (iterations, cutoff)
            for iterations in FISTA_ITERATIONS
            for cutoff in PARTITION_CUTOFFS
        )
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "gemma_sv_residual_qualified_proxy_frontier",
        "status": "running",
        "claim_scope": "residual-qualified approximation; not a certificate",
        "manifest": {
            "path": str(args.manifest),
            "sha256": hashlib.sha256(raw_manifest).hexdigest(),
        },
        "split": {
            "calibration_indices": list(CALIBRATION_INDICES),
            "validation_indices": list(VALIDATION_INDICES),
            "locked_before_evaluation": True,
        },
        "grid": {
            "fista_iterations": list(FISTA_ITERATIONS),
            "partition_cutoffs": list(PARTITION_CUTOFFS),
            "residual_cutoffs": list(RESIDUAL_CUTOFFS),
        },
        "calibration": {"observations": []},
        "validation": {"observations": []},
    }
    _atomic_write(args.out, report)
    calibration = _run_group(
        matrix,
        records,
        calibration_indices,
        calibration_configs,
        args=args,
        report=report,
        section="calibration",
        output=args.out,
    )
    selection = select_calibration_policy(calibration)
    report["calibration"]["selection"] = selection
    policy = selection["selected_policy"]
    if policy is None or args.smoke:
        report["status"] = (
            "smoke_completed" if args.smoke else "negative_calibration_frontier"
        )
        _atomic_write(args.out, report)
        return 0

    validation_config = (
        (
            int(policy["fista_iterations"]),
            float(policy["partition_cutoff"]),
        ),
    )
    validation = _run_group(
        matrix,
        records,
        VALIDATION_INDICES,
        validation_config,
        args=args,
        report=report,
        section="validation",
        output=args.out,
    )
    report["validation"]["locked_policy_result"] = evaluate_locked_policy(
        validation,
        policy,
    )
    report["status"] = "completed"
    _atomic_write(args.out, report)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

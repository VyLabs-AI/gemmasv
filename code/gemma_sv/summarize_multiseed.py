"""Deterministically aggregate matched Gemma recovery and deletion artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence


LOG_FLOOR = 1e-300
MIN_SEED_COUNT = 3
ADMISSION_THRESHOLDS = {
    "minimum_answer_lift_nats": 0.05,
    "minimum_secret_lift_nats": 0.05,
    "maximum_first_token_rank": 10,
    "all_fields_must_pass": True,
    "fixed_c_all_boundaries_must_be_feasible": True,
}

# Two-sided 95% Student-t critical values.  The experiment's primary case is
# n=3 (df=2); the full table keeps optional subsets and extensions dependency-free.
_T95 = {
    1: 12.706204736432095,
    2: 4.302652729696142,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.4469118487916806,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
    11: 2.200985160082949,
    12: 2.1788128296634177,
    13: 2.160368656461013,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778331806,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727662,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.042272456301238,
}

_EXACT_KL_KEYS = {
    "kl_exact_vs_refit_nats",
    "exact_vs_refit_kl_nats",
    "output_kl_exact_vs_refit_nats",
    "kl_forget_vs_refit_nats",
}
_DECAY_KL_KEYS = {
    "kl_decay_vs_refit_nats",
    "decay_vs_refit_kl_nats",
    "output_kl_decay_vs_refit_nats",
}
_PROXY_KL_KEYS = {
    "kl_proxy_vs_exact_nats",
    "proxy_vs_exact_kl_nats",
    "output_kl_proxy_vs_exact_nats",
    "proxy_vs_exact_output_kl_nats",
}


class AggregationError(ValueError):
    """An input artifact is incomplete or internally inconsistent."""


class PairingError(AggregationError):
    """Recovery/control artifacts do not describe the same seeded run."""


@dataclass(frozen=True)
class Artifact:
    path: Path
    payload: Mapping[str, Any]
    sha256: str
    seed: int


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AggregationError(f"{label} must be finite")
    return result


def _t_critical_95(sample_size: int) -> float:
    if sample_size < 2:
        raise AggregationError("a Student-t interval requires at least two values")
    degrees = sample_size - 1
    if degrees in _T95:
        return _T95[degrees]

    # Cornish-Fisher expansion around the standard-normal quantile.  The exact
    # small-df values above cover all three-seed analyses.
    z = NormalDist().inv_cdf(0.975)
    d = float(degrees)
    return (
        z
        + (z**3 + z) / (4.0 * d)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * d**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * d**3)
    )


def student_t_ci95(values: Sequence[float]) -> dict[str, float | int | None | str]:
    """Return a two-sided 95% CI for the mean, using sample variance."""
    clean = [_as_number(value, "CI value") for value in values]
    if not clean:
        raise AggregationError("cannot summarize an empty value sequence")
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        return {
            "n": 1,
            "mean": mean,
            "lower": None,
            "upper": None,
            "half_width": None,
            "t_critical": None,
            "method": "95% Student-t across seeds; undefined for n=1",
        }
    critical = _t_critical_95(len(clean))
    half_width = critical * statistics.stdev(clean) / math.sqrt(len(clean))
    return {
        "n": len(clean),
        "mean": mean,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "half_width": half_width,
        "t_critical": critical,
        "method": "two-sided 95% Student-t across seeds",
    }


def floor_safe_log10(value: float, floor: float = LOG_FLOOR) -> float:
    """Take log10 after flooring nonnegative numerical-zero observations."""
    number = _as_number(value, "log metric")
    safe_floor = _as_number(floor, "log floor")
    if number < 0:
        raise AggregationError("log metrics must be nonnegative")
    if safe_floor <= 0:
        raise AggregationError("log floor must be positive")
    return math.log10(max(number, safe_floor))


def _finite_pow10(exponent: float) -> tuple[float, bool]:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    maximum_exponent = math.log10(maximum)
    if exponent > maximum_exponent:
        return maximum, True
    return 10.0**exponent, False


def floor_safe_log_summary(
    values: Sequence[float],
    floor: float = LOG_FLOOR,
) -> dict[str, float | int]:
    """Summarize raw numerical-floor values without producing -inf or NaN."""
    clean = [_as_number(value, "log metric") for value in values]
    if not clean:
        raise AggregationError("cannot summarize an empty log metric")
    if any(value < 0 for value in clean):
        raise AggregationError("log metrics must be nonnegative")
    logs = [floor_safe_log10(value, floor) for value in clean]
    mean_log = statistics.fmean(logs)
    return {
        "n": len(clean),
        "floor": floor,
        "floored_count": sum(value < floor for value in clean),
        "mean_log10": mean_log,
        "geometric_mean": 10.0**mean_log,
        "worst": max(clean),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _at(payload: Mapping[str, Any], *path: str) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _seed_from_payload(
    payload: Mapping[str, Any],
    source: Path | None = None,
) -> int:
    candidates = [
        payload.get("seed"),
        payload.get("run_seed"),
        payload.get("training_seed"),
        _at(payload, "config", "run_seed"),
        _at(payload, "config", "training_seed"),
        _at(payload, "config", "seed"),
        _at(payload, "provenance", "run_seed"),
        _at(payload, "provenance", "training_seed"),
        _at(payload, "provenance", "seed"),
    ]
    seeds = set()
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, bool):
            raise AggregationError("seed must be an integer")
        try:
            seed = int(value)
        except (TypeError, ValueError) as error:
            raise AggregationError(f"invalid seed {value!r}") from error
        if str(seed) != str(value) and not isinstance(value, int):
            raise AggregationError(f"invalid seed {value!r}")
        seeds.add(seed)
    if len(seeds) > 1:
        raise AggregationError(f"conflicting seed metadata: {sorted(seeds)}")
    if seeds:
        return next(iter(seeds))

    if source is not None:
        matches = {
            int(match)
            for match in re.findall(
                r"(?:^|[^a-z0-9])seed[-_]?(\d+)(?:[^0-9]|$)",
                source.as_posix().lower(),
            )
        }
        if len(matches) == 1:
            return next(iter(matches))
    raise AggregationError("artifact is missing an unambiguous training seed")


def _walk(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            yield path, name, child
            yield from _walk(child, path + (name,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, path + (str(index),))


def _stage2_fingerprint(payload: Mapping[str, Any]) -> str | None:
    exact_names = {
        "stage2_batch_fingerprint",
        "stage_2_batch_fingerprint",
        "stage2_stream_fingerprint",
        "stage2_batches_sha256",
        "stage2_fingerprint",
    }
    found = set()
    for candidate in (
        payload.get("stage2_stream"),
        _at(payload, "provenance", "stage2_stream"),
    ):
        if isinstance(candidate, Mapping):
            found.add(_canonical(candidate))
    for path, key, value in _walk(payload):
        normalized = key.lower().replace("-", "_")
        is_nested = normalized == "batch_fingerprint" and any(
            "stage2" in part.lower().replace("-", "_")
            or "stage_2" in part.lower().replace("-", "_")
            for part in path
        )
        if normalized in exact_names or is_nested:
            found.add(_canonical(value))
    if len(found) > 1:
        raise PairingError("artifact contains conflicting stage-2 batch fingerprints")
    return next(iter(found)) if found else None


def _first_config_value(
    sources: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
) -> Any:
    for source in sources:
        for alias in aliases:
            if alias in source and source[alias] is not None:
                return source[alias]
    return None


def _stage2_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    sources: list[Mapping[str, Any]] = []
    for candidate in (
        payload.get("stage2_config"),
        payload.get("stage2_batch_config"),
        _at(payload, "stage2", "config"),
        _at(payload, "provenance", "stage2_config"),
        payload.get("config"),
        payload,
    ):
        if isinstance(candidate, Mapping):
            sources.append(candidate)

    aliases = {
        "model": ("model", "model_id"),
        "batch": ("batch", "batch_size"),
        "seq_len": ("seq_len", "sequence_length"),
        "stage2_steps": ("stage2_steps", "steps", "num_steps"),
        "lr2": ("lr2", "learning_rate", "lr"),
        "rank": ("rank", "lora_rank"),
        "dataset": ("dataset", "train_dataset"),
        "dataset_revision": (
            "dataset_revision",
            "data_revision",
            "fineweb_revision",
        ),
        "model_revision": ("model_revision",),
        "data_seed": ("data_seed",),
        "init_seed": ("init_seed", "adapter_init_seed", "lora_seed"),
        "lora_alpha": ("lora_alpha",),
        "lora_dropout": ("lora_dropout",),
        "target_modules": ("target_modules",),
        "optimizer": ("optimizer",),
        "dtype": ("dtype", "train_dtype"),
    }
    normalized = {}
    for name, names in aliases.items():
        value = _first_config_value(sources, names)
        if value is not None:
            normalized[name] = value

    required = {"model", "batch", "seq_len", "stage2_steps", "lr2", "rank"}
    missing = sorted(required - normalized.keys())
    if missing:
        raise PairingError(
            "stage-2 config is missing required fields: " + ", ".join(missing)
        )
    return normalized


def validate_recovery_control_pair(
    recovery: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the independent-unit seed, exact batch stream, and stage-2 recipe."""
    recovery_seed = _seed_from_payload(recovery)
    control_seed = _seed_from_payload(control)
    if recovery_seed != control_seed:
        raise PairingError(
            f"seed mismatch: recovery={recovery_seed}, control={control_seed}"
        )

    recovery_fingerprint = _stage2_fingerprint(recovery)
    control_fingerprint = _stage2_fingerprint(control)
    if not recovery_fingerprint or not control_fingerprint:
        raise PairingError("both arms must record a stage-2 batch fingerprint")
    if recovery_fingerprint != control_fingerprint:
        raise PairingError(
            f"stage-2 batch fingerprint mismatch for seed {recovery_seed}"
        )

    recovery_config = _stage2_config(recovery)
    control_config = _stage2_config(control)
    if _canonical(recovery_config) != _canonical(control_config):
        raise PairingError(f"stage-2 config mismatch for seed {recovery_seed}")
    return {
        "seed": recovery_seed,
        "stage2_batch_fingerprint": json.loads(recovery_fingerprint),
        "stage2_config": recovery_config,
    }


def _first_number(
    payload: Mapping[str, Any],
    paths: Sequence[tuple[str, ...]],
    label: str,
) -> float:
    for path in paths:
        value = _at(payload, *path)
        if value is not None:
            return _as_number(value, label)
    raise AggregationError(f"artifact is missing {label}")


def _recovered_ppl(payload: Mapping[str, Any]) -> float:
    return _first_number(
        payload,
        (
            ("ppl_final",),
            ("ppl_recovered",),
            ("recovered_ppl",),
            ("metrics", "ppl_final"),
            ("metrics", "recovered_ppl"),
        ),
        "recovered perplexity",
    )


def _control_ppl(payload: Mapping[str, Any]) -> float:
    return _first_number(
        payload,
        (
            ("ppl_control",),
            ("control_ppl",),
            ("metrics", "ppl_control"),
            ("metrics", "control_ppl"),
        ),
        "control perplexity",
    )


def _load(path: Path) -> Artifact:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AggregationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise AggregationError(f"{path} must contain a JSON object")
    return Artifact(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
        seed=_seed_from_payload(payload, path),
    )


def _index(
    paths: Sequence[Path],
    label: str,
    *,
    expected_seeds: set[int] | None = None,
    require_complete: bool = False,
) -> dict[int, Artifact]:
    indexed: dict[int, Artifact] = {}
    for path in sorted({Path(path) for path in paths}, key=lambda item: item.as_posix()):
        artifact = _load(path)
        if artifact.seed in indexed:
            raise AggregationError(f"duplicate {label} artifact for seed {artifact.seed}")
        indexed[artifact.seed] = artifact
    if expected_seeds is not None:
        extra = sorted(set(indexed) - expected_seeds)
        missing = sorted(expected_seeds - set(indexed))
        if extra or (require_complete and missing):
            raise AggregationError(
                f"{label} seed coverage mismatch; missing={missing}, extra={extra}"
            )
    return indexed


def _metric_values(
    payload: Mapping[str, Any],
    names: set[str],
    label: str,
) -> list[float]:
    values = []
    for path, key, value in _walk(payload):
        path_parts = {part.lower() for part in path}
        if key.lower() not in names or path_parts.intersection(
            {"summary", "directions", "metric_definitions", "semantics"}
        ):
            continue
        number = _as_number(value, label)
        if number < 0:
            raise AggregationError(f"{label} must be nonnegative")
        values.append(number)
    return values


def _fallback_count(payload: Mapping[str, Any]) -> int:
    for key_name in ("decrement_fallbacks", "n_fallback", "total_fallbacks"):
        values = []
        for path, key, value in _walk(payload):
            if key.lower() != key_name or "summary" in {
                part.lower() for part in path
            }:
                continue
            number = _as_number(value, "fallback count")
            if number < 0 or not number.is_integer():
                raise AggregationError("fallback counts must be nonnegative integers")
            values.append(int(number))
        if values:
            return sum(values)
    return 0


def _metric_by_seed(
    artifacts: Mapping[int, Artifact],
    names: set[str],
    label: str,
) -> dict[int, list[float]]:
    result = {}
    for seed, artifact in sorted(artifacts.items()):
        values = _metric_values(artifact.payload, names, label)
        if not values:
            raise AggregationError(
                f"{artifact.path} contains no recognized {label} observations"
            )
        result[seed] = values
    return result


def _seed_level_metric_summary(
    values_by_seed: Mapping[int, Sequence[float]],
    *,
    floor: float = LOG_FLOOR,
) -> dict[str, Any]:
    seed_rows = []
    all_values = []
    for seed, values in sorted(values_by_seed.items()):
        clean = [_as_number(value, "certificate metric") for value in values]
        if not clean:
            raise AggregationError(f"seed {seed} has no certificate observations")
        log_summary = floor_safe_log_summary(clean, floor)
        all_values.extend(clean)
        seed_rows.append(
            {
                "seed": seed,
                "n_observations": len(clean),
                "raw_arithmetic_mean": statistics.fmean(clean),
                "mean_log10": log_summary["mean_log10"],
                "geometric_mean": log_summary["geometric_mean"],
                "worst": max(clean),
                "floored_count": log_summary["floored_count"],
            }
        )

    raw_ci = student_t_ci95(
        [float(row["raw_arithmetic_mean"]) for row in seed_rows]
    )
    log_ci = student_t_ci95([float(row["mean_log10"]) for row in seed_rows])
    geometric_mean, mean_clipped = _finite_pow10(float(log_ci["mean"]))
    geometric_lower, lower_clipped = (
        (None, False)
        if log_ci["lower"] is None
        else _finite_pow10(float(log_ci["lower"]))
    )
    geometric_upper, upper_clipped = (
        (None, False)
        if log_ci["upper"] is None
        else _finite_pow10(float(log_ci["upper"]))
    )
    geometric = {
        "n": log_ci["n"],
        "mean": geometric_mean,
        "lower": geometric_lower,
        "upper": geometric_upper,
        "clipped_to_finite": {
            "mean": mean_clipped,
            "lower": lower_clipped,
            "upper": upper_clipped,
        },
        "method": "back-transform of seed-level mean-log10 Student-t CI",
    }
    return {
        "n_seeds": len(seed_rows),
        "n_observations": len(all_values),
        "log_floor": floor,
        "floored_observations": sum(
            int(row["floored_count"]) for row in seed_rows
        ),
        "raw_arithmetic": raw_ci
        | {"pooled_observation_mean": statistics.fmean(all_values)},
        "log10": log_ci,
        "geometric": geometric,
        "worst": max(all_values),
        "seed_rows": seed_rows,
    }


def _adapter_hash(payload: Mapping[str, Any]) -> str | None:
    candidates = (
        _at(payload, "provenance", "adapter", "content_sha256"),
        _at(payload, "provenance", "adapter", "sha256"),
        _at(payload, "adapter", "content_sha256"),
        payload.get("adapter_sha256"),
    )
    found = {str(value) for value in candidates if value}
    if len(found) > 1:
        raise PairingError("artifact contains conflicting adapter hashes")
    return next(iter(found)) if found else None


def _validate_auxiliary(
    artifact: Artifact,
    recovery: Artifact,
    label: str,
) -> None:
    if artifact.seed != recovery.seed:
        raise PairingError(
            f"{label} seed {artifact.seed} does not match recovery seed {recovery.seed}"
        )
    auxiliary_fingerprint = _stage2_fingerprint(artifact.payload)
    recovery_fingerprint = _stage2_fingerprint(recovery.payload)
    if (
        auxiliary_fingerprint
        and recovery_fingerprint
        and auxiliary_fingerprint != recovery_fingerprint
    ):
        raise PairingError(f"{label} stage-2 fingerprint mismatch for seed {artifact.seed}")
    auxiliary_hash = _adapter_hash(artifact.payload)
    recovery_hash = _adapter_hash(recovery.payload)
    if auxiliary_hash and recovery_hash and auxiliary_hash != recovery_hash:
        raise PairingError(f"{label} adapter hash mismatch for seed {artifact.seed}")


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return {}
        result[str(key)] = _as_number(item, str(key))
    return result


def _arm_mapping(
    payload: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, float]:
    for name in names:
        result = _numeric_mapping(payload.get(name))
        if result:
            return result
        result = _numeric_mapping(_at(payload, "metrics", name))
        if result:
            return result
    return {}


def _paired_cross_corpus(payload: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    candidates = [
        payload.get("cross_corpus"),
        payload.get("cross_corpus_ppl"),
        payload.get("ppl2"),
        payload,
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        recovered = _numeric_mapping(candidate.get("recovered"))
        control = _numeric_mapping(candidate.get("control"))
        if recovered and control:
            return recovered, control
    return {}, {}


def _paired_zero_shot(payload: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    candidates = [
        payload.get("zero_shot"),
        payload.get("zero_shot_tasks"),
        payload.get("tasks"),
        payload,
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        recovered, control = {}, {}
        for task, row in candidate.items():
            if not isinstance(row, Mapping):
                continue
            if "recovered" in row and "control" in row:
                recovered[str(task)] = _as_number(
                    row["recovered"], f"{task} recovered score"
                )
                control[str(task)] = _as_number(
                    row["control"], f"{task} control score"
                )
        if recovered:
            return recovered, control
    return {}, {}


def _merge_arm_values(
    current: dict[str, float],
    incoming: Mapping[str, float],
    label: str,
) -> None:
    for name, value in incoming.items():
        if name in current and current[name] != value:
            raise AggregationError(f"conflicting {label} value for {name}")
        current[name] = value


def _optional_deltas(
    seeds: Sequence[int],
    recoveries: Mapping[int, Artifact],
    controls: Mapping[int, Artifact],
    extras: Mapping[int, Artifact],
    *,
    kind: str,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    rows: dict[int, dict[str, float]] = {}
    for seed in seeds:
        if kind == "cross_corpus":
            recovered = _arm_mapping(
                recoveries[seed].payload,
                ("cross_corpus", "cross_corpus_ppl", "ppl2"),
            )
            control = _arm_mapping(
                controls[seed].payload,
                ("cross_corpus", "cross_corpus_ppl", "ppl2"),
            )
            if seed in extras:
                extra_recovered, extra_control = _paired_cross_corpus(
                    extras[seed].payload
                )
                _merge_arm_values(recovered, extra_recovered, "cross-corpus")
                _merge_arm_values(control, extra_control, "cross-corpus")
        else:
            recovered = _arm_mapping(
                recoveries[seed].payload,
                ("zero_shot", "zero_shot_tasks", "tasks"),
            )
            control = _arm_mapping(
                controls[seed].payload,
                ("zero_shot", "zero_shot_tasks", "tasks"),
            )
            if seed in extras:
                extra_recovered, extra_control = _paired_zero_shot(
                    extras[seed].payload
                )
                _merge_arm_values(recovered, extra_recovered, "zero-shot")
                _merge_arm_values(control, extra_control, "zero-shot")

        if not recovered and not control:
            continue
        if set(recovered) != set(control):
            raise AggregationError(
                f"{kind} recovered/control metric coverage differs for seed {seed}"
            )
        if kind == "cross_corpus":
            if any(recovered[name] <= 0 or control[name] <= 0 for name in recovered):
                raise AggregationError("cross-corpus perplexities must be positive")
            rows[seed] = {
                name: 100.0 * (recovered[name] / control[name] - 1.0)
                for name in sorted(recovered)
            }
        else:
            rows[seed] = {
                name: recovered[name] - control[name]
                for name in sorted(recovered)
            }

    names = sorted({name for values in rows.values() for name in values})
    summary = {}
    for name in names:
        present = [rows[seed][name] for seed in seeds if name in rows.get(seed, {})]
        summary[name] = student_t_ci95(present) | {
            "missing_seeds": [
                seed for seed in seeds if name not in rows.get(seed, {})
            ]
        }
    return rows, summary


def _ratio(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": float(numerator / denominator) if denominator else 0.0,
    }


def _validate_admission_thresholds(
    value: Any,
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise AggregationError(f"{label} is missing admission thresholds")
    expected_keys = set(ADMISSION_THRESHOLDS)
    observed_keys = set(value)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise AggregationError(
            f"{label} admission threshold keys differ from the frozen policy; "
            f"missing={missing}, extra={extra}"
        )
    for name, expected in ADMISSION_THRESHOLDS.items():
        observed = value[name]
        if isinstance(expected, bool):
            matches = isinstance(observed, bool) and observed is expected
        else:
            matches = (
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and float(observed) == float(expected)
            )
        if not matches:
            raise AggregationError(
                f"{label} admission threshold {name}={observed!r} "
                f"does not match the frozen value {expected!r}"
            )


def _recompute_admission_row(
    admission: Mapping[str, Any],
    *,
    label: str = "admission row",
) -> dict[str, Any]:
    """Recompute every frozen admission gate from one raw audit row."""

    _validate_admission_thresholds(admission.get("thresholds"), label=label)
    thresholds = ADMISSION_THRESHOLDS
    reasons: list[str] = []

    answer_lift = _as_number(
        admission.get("record_answer_lift_nats"),
        f"{label} record-answer lift",
    )
    answer_pass = answer_lift >= float(thresholds["minimum_answer_lift_nats"])
    if not answer_pass:
        reasons.append("record:answer_lift")

    fields = admission.get("deleted_fields")
    if not isinstance(fields, list) or not fields:
        raise AggregationError(f"{label} must contain deleted-field gate rows")
    field_passes = 0
    probe_ids: set[str] = set()
    for index, field in enumerate(fields):
        if not isinstance(field, Mapping):
            raise AggregationError(f"{label} deleted field {index} must be an object")
        probe_id = str(field.get("probe_id", ""))
        if not probe_id.startswith("deleted_field_") or probe_id in probe_ids:
            raise AggregationError(
                f"{label} has an invalid or duplicate deleted probe ID {probe_id!r}"
            )
        probe_ids.add(probe_id)
        lift = _as_number(
            field.get("secret_lift_nats"),
            f"{label} {probe_id} secret lift",
        )
        rank = _as_number(
            field.get("first_token_rank"),
            f"{label} {probe_id} first-token rank",
        )
        if rank < 1 or not rank.is_integer():
            raise AggregationError(
                f"{label} {probe_id} first-token rank must be a positive integer"
            )
        lift_pass = lift >= float(thresholds["minimum_secret_lift_nats"])
        rank_pass = rank <= int(thresholds["maximum_first_token_rank"])
        if not lift_pass:
            reasons.append(f"{probe_id}:secret_lift")
        if not rank_pass:
            reasons.append(f"{probe_id}:rank")
        field_passes += int(lift_pass and rank_pass)
    expected_probe_ids = {
        f"deleted_field_{index}" for index in range(len(fields))
    }
    if probe_ids != expected_probe_ids:
        raise AggregationError(
            f"{label} deleted probe IDs differ from the expected sequence"
        )

    retained = admission.get("retained_field")
    if not isinstance(retained, Mapping):
        raise AggregationError(f"{label} must contain a retained-neighbor gate row")
    retained_probe = str(retained.get("probe_id", ""))
    if retained_probe != "retained_field":
        raise AggregationError(
            f"{label} has unexpected retained probe ID {retained_probe!r}"
        )
    retained_rank = _as_number(
        retained.get("first_token_rank"),
        f"{label} retained first-token rank",
    )
    if retained_rank < 1 or not retained_rank.is_integer():
        raise AggregationError(
            f"{label} retained first-token rank must be a positive integer"
        )
    retained_pass = retained_rank <= int(thresholds["maximum_first_token_rank"])
    if not retained_pass:
        reasons.append("retained_field:rank")

    boundaries = admission.get("fixed_c_feasibility")
    if not isinstance(boundaries, list) or not boundaries:
        raise AggregationError(f"{label} must contain fixed-C boundary diagnostics")
    feasible_boundaries = 0
    boundary_starts: set[int] = set()
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise AggregationError(
                f"{label} fixed-C boundary {index} must be an object"
            )
        start = _as_number(
            boundary.get("start"),
            f"{label} fixed-C boundary {index} start",
        )
        if start < 1 or not start.is_integer() or int(start) in boundary_starts:
            raise AggregationError(
                f"{label} fixed-C boundary starts must be unique positive integers"
            )
        boundary_starts.add(int(start))
        stored_feasible = boundary.get("feasible")
        if not isinstance(stored_feasible, bool):
            raise AggregationError(
                f"{label} fixed-C boundary {index} feasibility must be boolean"
            )
        retained_capacity = _as_number(
            boundary.get("retained_capacity"),
            f"{label} fixed-C boundary {index} retained capacity",
        )
        recomputed_feasible = retained_capacity >= 1.0 - 1e-12
        if stored_feasible is not recomputed_feasible:
            raise AggregationError(
                f"{label} fixed-C boundary {int(start)} stored feasibility "
                "differs from retained capacity"
            )
        feasible_boundaries += int(recomputed_feasible)
    fixed_c_pass = feasible_boundaries == len(boundaries)
    if not fixed_c_pass:
        reasons.append("fixed_c:infeasible")

    recomputed_reasons = sorted(set(reasons))
    stored_reasons = admission.get("reasons")
    if not isinstance(stored_reasons, list) or not all(
        isinstance(reason, str) for reason in stored_reasons
    ):
        raise AggregationError(f"{label} stored reason codes must be a string list")
    if stored_reasons != sorted(set(stored_reasons)):
        raise AggregationError(
            f"{label} stored reason codes must be sorted and duplicate-free"
        )
    if stored_reasons != recomputed_reasons:
        raise AggregationError(
            f"{label} stored reason codes differ from recomputed gates: "
            f"stored={stored_reasons}, recomputed={recomputed_reasons}"
        )

    recomputed_status = "rejected" if recomputed_reasons else "admitted"
    if admission.get("status") != recomputed_status:
        raise AggregationError(
            f"{label} stored status {admission.get('status')!r} does not match "
            f"recomputed status {recomputed_status!r}"
        )

    target_only_pass = answer_pass and field_passes == len(fields)
    return {
        "joint_target_retained_pass": (
            target_only_pass and retained_pass and fixed_c_pass
        ),
        "target_only_pass": target_only_pass,
        "deleted_fields_passed": field_passes,
        "deleted_fields_total": len(fields),
        "retained_neighbor_pass": retained_pass,
        "fixed_c_context_pass": fixed_c_pass,
        "fixed_c_boundaries_passed": feasible_boundaries,
        "fixed_c_boundaries_total": len(boundaries),
        "recomputed_reasons": recomputed_reasons,
    }


def _admission_decomposition(
    seeds: Sequence[int],
    audits: Mapping[int, Artifact],
) -> dict[str, Any]:
    totals = {
        "contexts": 0,
        "joint": 0,
        "target_only": 0,
        "fields_passed": 0,
        "fields_total": 0,
        "retained": 0,
        "fixed_contexts": 0,
        "fixed_boundaries_passed": 0,
        "fixed_boundaries_total": 0,
    }
    seed_rows = []
    for seed in seeds:
        payload = audits[seed].payload
        runs = payload.get("runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise AggregationError(
                f"{audits[seed].path} must contain exactly one adapter run"
            )
        run = runs[0]
        records = run.get("records") if isinstance(run, Mapping) else None
        summary = run.get("summary") if isinstance(run, Mapping) else None
        if not isinstance(records, list) or not records:
            raise AggregationError(f"{audits[seed].path} has no raw audit records")
        if not isinstance(summary, Mapping):
            raise AggregationError(f"{audits[seed].path} has no run summary")

        current = {name: 0 for name in totals}
        rejected = []
        record_ids: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise AggregationError(
                    f"{audits[seed].path} record {index} must be an object"
                )
            record_id = str(record.get("record_id", f"index-{index}"))
            if record_id in record_ids:
                raise AggregationError(
                    f"{audits[seed].path} has duplicate record ID {record_id!r}"
                )
            record_ids.add(record_id)
            admission = record.get("admission")
            if not isinstance(admission, Mapping):
                raise AggregationError(
                    f"{audits[seed].path} record {record_id} has no admission row"
                )
            components = _recompute_admission_row(
                admission,
                label=f"seed {seed} record {record_id}",
            )
            current["contexts"] += 1
            current["joint"] += int(components["joint_target_retained_pass"])
            current["target_only"] += int(components["target_only_pass"])
            current["fields_passed"] += int(components["deleted_fields_passed"])
            current["fields_total"] += int(components["deleted_fields_total"])
            current["retained"] += int(components["retained_neighbor_pass"])
            current["fixed_contexts"] += int(components["fixed_c_context_pass"])
            current["fixed_boundaries_passed"] += int(
                components["fixed_c_boundaries_passed"]
            )
            current["fixed_boundaries_total"] += int(
                components["fixed_c_boundaries_total"]
            )
            if components["recomputed_reasons"]:
                rejected.append(
                    {
                        "record_id": record_id,
                        "reasons": components["recomputed_reasons"],
                    }
                )

        attempted = int(summary.get("attempted_records", -1))
        admitted = int(summary.get("admitted_records", -1))
        stored_rejected = summary.get("rejected_records")
        if attempted != current["contexts"]:
            raise AggregationError(
                f"{audits[seed].path} stored attempted-record count does not "
                "match raw records"
            )
        if admitted != current["joint"]:
            raise AggregationError(
                f"{audits[seed].path} stored admitted-record count does not "
                "match recomputed joint admission"
            )
        if stored_rejected != rejected:
            raise AggregationError(
                f"{audits[seed].path} stored rejected-record reasons do not "
                "match recomputed gates"
            )
        for name in totals:
            totals[name] += current[name]
        seed_rows.append(
            {
                "seed": seed,
                "joint_target_and_retained_whole_record": _ratio(
                    current["joint"], current["contexts"]
                ),
                "target_only_whole_record": _ratio(
                    current["target_only"], current["contexts"]
                ),
                "deleted_fields": _ratio(
                    current["fields_passed"], current["fields_total"]
                ),
                "retained_neighbor_availability": _ratio(
                    current["retained"], current["contexts"]
                ),
                "fixed_c_contexts": _ratio(
                    current["fixed_contexts"], current["contexts"]
                ),
                "fixed_c_boundaries": _ratio(
                    current["fixed_boundaries_passed"],
                    current["fixed_boundaries_total"],
                ),
            }
        )

    return {
        "policy": (
            "decomposition of the unchanged frozen admission gates; these "
            "components do not revise inclusion criteria"
        ),
        "thresholds": dict(ADMISSION_THRESHOLDS),
        "threshold_keys_verified": True,
        "fixed_c_retained_capacity_verified": True,
        "stored_reason_codes_verified": True,
        "joint_target_and_retained_whole_record": _ratio(
            totals["joint"], totals["contexts"]
        ),
        "target_only_whole_record": _ratio(
            totals["target_only"], totals["contexts"]
        ),
        "deleted_fields": _ratio(
            totals["fields_passed"], totals["fields_total"]
        ),
        "retained_neighbor_availability": _ratio(
            totals["retained"], totals["contexts"]
        ),
        "fixed_c_feasibility": {
            "contexts": _ratio(totals["fixed_contexts"], totals["contexts"]),
            "boundaries": _ratio(
                totals["fixed_boundaries_passed"],
                totals["fixed_boundaries_total"],
            ),
        },
        "seed_rows": seed_rows,
    }


def _deletion_baseline_summary(
    seeds: Sequence[int],
    audits: Mapping[int, Artifact],
) -> dict[str, Any]:
    """Aggregate each audit method with training seed as the CI unit."""

    seed_methods: dict[int, Mapping[str, Any]] = {}
    seed_model_storage: dict[int, dict[str, float]] = {}
    seed_admission: dict[int, dict[str, Any]] = {}
    seed_exact_execution: dict[int, dict[str, int]] = {}
    excluded = None
    for seed in seeds:
        payload = audits[seed].payload
        runs = payload.get("runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise AggregationError(
                f"{audits[seed].path} must contain exactly one adapter run"
            )
        run = runs[0]
        summary = run.get("summary") if isinstance(run, Mapping) else None
        methods = summary.get("methods") if isinstance(summary, Mapping) else None
        if not isinstance(methods, Mapping) or not methods:
            raise AggregationError(
                f"{audits[seed].path} has no deletion method summary"
            )
        seed_methods[seed] = methods
        attempted = int(summary.get("attempted_records", 0))
        admitted = int(summary.get("admitted_records", 0))
        seed_admission[seed] = {
            "attempted_records": attempted,
            "admitted_records": admitted,
            "admission_rate": admitted / attempted if attempted else 0.0,
            "rejected_records": summary.get("rejected_records", []),
        }
        records = run.get("records")
        if isinstance(records, list):
            execution = {
                "head_gate_solves": 0,
                "decrement_fallbacks": 0,
                "completed_records": 0,
            }
            for record in records:
                precision = (
                    record.get("prefill_once_precision_audit")
                    if isinstance(record, Mapping)
                    else None
                )
                if not isinstance(precision, Mapping) or precision.get(
                    "status"
                ) != "completed":
                    continue
                diagnostics = precision.get("solver_diagnostics")
                if not isinstance(diagnostics, Mapping):
                    raise AggregationError(
                        f"{audits[seed].path} completed precision audit lacks "
                        "solver diagnostics"
                    )
                solves = int(diagnostics.get("head_gate_solves", -1))
                fallbacks = int(diagnostics.get("decrement_fallbacks", -1))
                if solves < 0 or not 0 <= fallbacks <= solves:
                    raise AggregationError(
                        f"{audits[seed].path} has invalid exact-path counts"
                    )
                execution["head_gate_solves"] += solves
                execution["decrement_fallbacks"] += fallbacks
                execution["completed_records"] += 1
            if execution["completed_records"]:
                seed_exact_execution[seed] = execution
        tensor_storage = run.get("tensor_storage")
        if isinstance(tensor_storage, Mapping):
            model_storage = tensor_storage.get("model_parameters_and_buffers")
            adapter_storage = tensor_storage.get("adapter_named_tensors")
            if isinstance(model_storage, Mapping) and isinstance(
                adapter_storage, Mapping
            ):
                seed_model_storage[seed] = {
                    "model_parameters_and_buffers_bytes": _as_number(
                        model_storage.get("deduplicated_tensor_storage_bytes"),
                        "model tensor storage",
                    ),
                    "adapter_named_tensor_bytes": _as_number(
                        adapter_storage.get("deduplicated_tensor_storage_bytes"),
                        "adapter tensor storage",
                    ),
                }
        current_excluded = payload.get("excluded_methods")
        if current_excluded is not None:
            canonical = _canonical(current_excluded)
            if excluded is None:
                excluded = canonical
            elif excluded != canonical:
                raise AggregationError(
                    "deletion audits disagree on excluded-method compatibility"
                )

    method_ids = sorted(
        {
            str(method_id)
            for methods in seed_methods.values()
            for method_id in methods
        }
    )
    result: dict[str, Any] = {
        "ci_unit": "matched training seed",
        "methods": {},
    }
    if excluded is not None:
        result["excluded_methods"] = json.loads(excluded)
    result["admission"] = {
        "seed_rows": [
            {"seed": seed, **seed_admission[seed]} for seed in seeds
        ],
        "admission_rate": student_t_ci95(
            [seed_admission[seed]["admission_rate"] for seed in seeds]
        ),
        "decomposition": _admission_decomposition(seeds, audits),
    }
    if seed_model_storage:
        result["model_storage"] = {
            "seed_rows": [
                {"seed": seed, **seed_model_storage[seed]}
                for seed in seeds
                if seed in seed_model_storage
            ],
            "model_parameters_and_buffers_bytes": student_t_ci95(
                [
                    seed_model_storage[seed][
                        "model_parameters_and_buffers_bytes"
                    ]
                    for seed in seeds
                    if seed in seed_model_storage
                ]
            ),
            "adapter_named_tensor_bytes": student_t_ci95(
                [
                    seed_model_storage[seed]["adapter_named_tensor_bytes"]
                    for seed in seeds
                    if seed in seed_model_storage
                ]
            ),
        }
    if seed_exact_execution:
        missing = sorted(set(seeds) - set(seed_exact_execution))
        if missing:
            raise AggregationError(
                "exact-path execution diagnostics are missing for seeds "
                + ", ".join(str(seed) for seed in missing)
            )
        total_solves = sum(
            seed_exact_execution[seed]["head_gate_solves"] for seed in seeds
        )
        total_fallbacks = sum(
            seed_exact_execution[seed]["decrement_fallbacks"] for seed in seeds
        )
        result["exact_path_execution"] = {
            "head_gate_solves": total_solves,
            "decrement_successes": total_solves - total_fallbacks,
            "exact_refit_fallbacks": total_fallbacks,
            "fallback_rate": total_fallbacks / total_solves if total_solves else 0.0,
            "fallback_semantics": (
                "a failed decrement gate executes the same float64 retained-key "
                "refit used as the exact reference"
            ),
            "seed_rows": [
                {"seed": seed, **seed_exact_execution[seed]} for seed in seeds
            ],
            "plotted_update_timing_scope": (
                "shared construction and audit of decrement plus fixed-C refit "
                "states; not pure decrement latency"
            ),
        }
    for method_id in method_ids:
        seed_rows = []
        metric_names: set[str] = set()
        for seed in seeds:
            method = seed_methods[seed].get(method_id)
            if not isinstance(method, Mapping):
                raise AggregationError(
                    f"seed {seed} audit is missing method {method_id}"
                )
            metrics = {
                str(name): _as_number(value, f"{method_id}.{name}")
                for name, value in method.items()
                if str(name).startswith("mean_")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            if not metrics:
                raise AggregationError(
                    f"seed {seed} method {method_id} has no mean metrics"
                )
            metric_names.update(metrics)
            seed_rows.append(
                {
                    "seed": seed,
                    "completed_records": int(method.get("completed_records", 0)),
                    "failed_records": int(method.get("failed_records", 0)),
                    "metrics": metrics,
                }
            )

        metric_summary = {}
        for metric in sorted(metric_names):
            values = [
                row["metrics"][metric]
                for row in seed_rows
                if metric in row["metrics"]
            ]
            metric_summary[metric] = student_t_ci95(values) | {
                "missing_seeds": [
                    row["seed"]
                    for row in seed_rows
                    if metric not in row["metrics"]
                ]
            }
        result["methods"][method_id] = {
            "seed_rows": seed_rows,
            "metrics": metric_summary,
        }
    return result


def _source(artifact: Artifact) -> dict[str, str]:
    return {"path": artifact.path.as_posix(), "sha256": artifact.sha256}


def summarize_paths(
    recovery_paths: Sequence[Path],
    control_paths: Sequence[Path],
    *,
    certificate_paths: Sequence[Path] = (),
    audit_paths: Sequence[Path] = (),
    cross_corpus_paths: Sequence[Path] = (),
    zero_shot_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Build a deterministic matched-seed report from explicit artifact paths."""
    recoveries = _index(recovery_paths, "recovery")
    controls = _index(control_paths, "control")
    if len(recoveries) < MIN_SEED_COUNT or len(controls) < MIN_SEED_COUNT:
        raise AggregationError(
            f"at least {MIN_SEED_COUNT} seed-specific recovery and control artifacts "
            "are required"
        )
    seeds = sorted(recoveries)
    if set(controls) != set(recoveries):
        raise PairingError(
            f"recovery/control seed sets differ: {seeds} vs {sorted(controls)}"
        )
    expected = set(seeds)
    certificates = _index(
        certificate_paths,
        "certificate",
        expected_seeds=expected,
        require_complete=bool(certificate_paths),
    )
    audits = _index(
        audit_paths,
        "audit",
        expected_seeds=expected,
        require_complete=bool(audit_paths),
    )
    cross_corpus = _index(
        cross_corpus_paths,
        "cross-corpus",
        expected_seeds=expected,
    )
    zero_shot = _index(
        zero_shot_paths,
        "zero-shot",
        expected_seeds=expected,
    )

    seed_rows = []
    recovered_values, control_values, utility_values = [], [], []
    for seed in seeds:
        recovery = recoveries[seed]
        control = controls[seed]
        pairing = validate_recovery_control_pair(
            recovery.payload,
            control.payload,
        )
        recovered_ppl = _recovered_ppl(recovery.payload)
        control_ppl = _control_ppl(control.payload)
        if recovered_ppl <= 0 or control_ppl <= 0:
            raise AggregationError("perplexities must be positive")
        utility = 100.0 * (recovered_ppl / control_ppl - 1.0)
        recovered_values.append(recovered_ppl)
        control_values.append(control_ppl)
        utility_values.append(utility)
        seed_rows.append(
            {
                "seed": seed,
                "recovered_ppl": recovered_ppl,
                "control_ppl": control_ppl,
                "paired_utility_cost_percent": utility,
                "stage2_batch_fingerprint": pairing["stage2_batch_fingerprint"],
                "stage2_config": pairing["stage2_config"],
                "sources": {
                    "recovery": _source(recovery),
                    "control": _source(control),
                },
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "evaluation": "gemma_sv_matched_seed_summary",
        "seed_count": len(seeds),
        "seeds": seeds,
        "ci_unit": "matched training seed",
        "seed_rows": seed_rows,
        "perplexity": {
            "recovered": student_t_ci95(recovered_values),
            "control": student_t_ci95(control_values),
            "paired_utility_cost_percent": student_t_ci95(utility_values),
        },
    }

    if certificates:
        for seed in seeds:
            _validate_auxiliary(certificates[seed], recoveries[seed], "certificate")
        exact = _metric_by_seed(
            certificates,
            _EXACT_KL_KEYS,
            "exact-vs-refit KL",
        )
        decay = _metric_by_seed(
            certificates,
            _DECAY_KL_KEYS,
            "decay-vs-refit KL",
        )
        fallbacks = {
            seed: _fallback_count(certificates[seed].payload) for seed in seeds
        }
        report["certificate"] = {
            "exact_vs_refit_kl_nats": _seed_level_metric_summary(exact),
            "decay_vs_refit_kl_nats": _seed_level_metric_summary(decay),
            "fallbacks": {
                "total": sum(fallbacks.values()),
                "seeds_with_fallback": sum(value > 0 for value in fallbacks.values()),
                "by_seed": [
                    {"seed": seed, "count": fallbacks[seed]} for seed in seeds
                ],
            },
        }
        for row in seed_rows:
            seed = int(row["seed"])
            row["certificate"] = {
                "exact_vs_refit_raw_mean": statistics.fmean(exact[seed]),
                "exact_vs_refit_worst": max(exact[seed]),
                "decay_vs_refit_raw_mean": statistics.fmean(decay[seed]),
                "decay_vs_refit_worst": max(decay[seed]),
                "fallbacks": fallbacks[seed],
            }
            row["sources"]["certificate"] = _source(certificates[seed])

    if audits:
        for seed in seeds:
            _validate_auxiliary(audits[seed], recoveries[seed], "audit")
        proxy = _metric_by_seed(
            audits,
            _PROXY_KL_KEYS,
            "proxy-vs-exact KL",
        )
        audit_exact = _metric_by_seed(
            audits,
            _EXACT_KL_KEYS,
            "prefill-once exact-vs-refit KL",
        )
        audit_decay = _metric_by_seed(
            audits,
            _DECAY_KL_KEYS,
            "prefill-once decay-vs-refit KL",
        )
        report["audit"] = {
            "exact_vs_refit_kl_nats": _seed_level_metric_summary(
                audit_exact
            ),
            "proxy_vs_exact_kl_nats": _seed_level_metric_summary(proxy),
            "decay_vs_refit_kl_nats": _seed_level_metric_summary(
                audit_decay
            ),
        }
        report["deletion_baselines"] = _deletion_baseline_summary(
            seeds,
            audits,
        )
        for row in seed_rows:
            seed = int(row["seed"])
            row["audit"] = {
                "exact_vs_refit_raw_mean": statistics.fmean(
                    audit_exact[seed]
                ),
                "exact_vs_refit_worst": max(audit_exact[seed]),
                "proxy_vs_exact_raw_mean": statistics.fmean(proxy[seed]),
                "proxy_vs_exact_worst": max(proxy[seed]),
                "decay_vs_refit_raw_mean": statistics.fmean(
                    audit_decay[seed]
                ),
                "decay_vs_refit_worst": max(audit_decay[seed]),
            }
            row["sources"]["audit"] = _source(audits[seed])

    cross_rows, cross_summary = _optional_deltas(
        seeds,
        recoveries,
        controls,
        cross_corpus,
        kind="cross_corpus",
    )
    if cross_rows:
        cross_seed_means = {
            seed: statistics.fmean(cross_rows[seed].values())
            for seed in seeds
            if seed in cross_rows
        }
        report["cross_corpus"] = {
            "unit": "paired recovered-vs-control PPL cost, percent",
            "seed_rows": [
                {
                    "seed": seed,
                    "deltas_percent": cross_rows[seed],
                    "mean_delta_percent": cross_seed_means[seed],
                }
                for seed in seeds
                if seed in cross_rows
            ],
            "summary": cross_summary,
            "mean_across_corpora": student_t_ci95(
                [cross_seed_means[seed] for seed in seeds]
            ),
        }
        for row in seed_rows:
            seed = int(row["seed"])
            if seed in cross_rows:
                row["cross_corpus_deltas_percent"] = cross_rows[seed]
                if seed in cross_corpus:
                    row["sources"]["cross_corpus"] = _source(cross_corpus[seed])

    task_rows, task_summary = _optional_deltas(
        seeds,
        recoveries,
        controls,
        zero_shot,
        kind="zero_shot",
    )
    if task_rows:
        task_seed_means = {
            seed: statistics.fmean(task_rows[seed].values())
            for seed in seeds
            if seed in task_rows
        }
        report["zero_shot"] = {
            "unit": "paired recovered-minus-control task score",
            "seed_rows": [
                {
                    "seed": seed,
                    "deltas": task_rows[seed],
                    "mean_delta": task_seed_means[seed],
                }
                for seed in seeds
                if seed in task_rows
            ],
            "summary": task_summary,
            "mean_across_tasks": student_t_ci95(
                [task_seed_means[seed] for seed in seeds]
            ),
        }
        for row in seed_rows:
            seed = int(row["seed"])
            if seed in task_rows:
                row["zero_shot_deltas"] = task_rows[seed]
                if seed in zero_shot:
                    row["sources"]["zero_shot"] = _source(zero_shot[seed])

    return report


def deterministic_json(report: Mapping[str, Any]) -> str:
    return json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=False,
    ) + "\n"


def _classify(path: Path, payload: Mapping[str, Any]) -> str | None:
    name = path.name.lower()
    if any(key in payload for key in ("ppl_control", "control_ppl")):
        return "control"
    if any(key in payload for key in ("ppl_final", "ppl_recovered", "recovered_ppl")):
        return "recovery"
    if _metric_values(payload, _PROXY_KL_KEYS, "proxy-vs-exact KL"):
        return "audit"
    if _metric_values(payload, _EXACT_KL_KEYS, "exact-vs-refit KL"):
        return "certificate"
    if _paired_cross_corpus(payload) != ({}, {}):
        return "cross_corpus"
    if _paired_zero_shot(payload) != ({}, {}):
        return "zero_shot"
    if "audit" in name:
        return "audit"
    if "certificate" in name:
        return "certificate"
    return None


def _discover(roots: Sequence[Path]) -> dict[str, list[Path]]:
    result = {
        "recovery": [],
        "control": [],
        "certificate": [],
        "audit": [],
        "cross_corpus": [],
        "zero_shot": [],
    }
    candidates = set()
    for root in roots:
        if root.is_file() and root.suffix == ".json":
            candidates.add(root)
        elif root.is_dir():
            candidates.update(root.rglob("*.json"))
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        category = _classify(path, payload)
        if category:
            result[category].append(path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        "--input-root",
        dest="roots",
        action="append",
        type=Path,
        default=[],
        help="discover seed-specific JSON artifacts recursively",
    )
    parser.add_argument(
        "--recovery",
        "--recovery-result",
        dest="recovery",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--control",
        "--control-result",
        dest="control",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--certificate",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--audit", action="append", type=Path, default=[])
    parser.add_argument(
        "--cross-corpus",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--zero-shot",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--out",
        default="-",
        help="deterministic JSON destination; '-' writes to stdout",
    )
    args = parser.parse_args(argv)

    roots = args.roots
    if not roots and not (args.recovery or args.control):
        roots = [Path("outputs/gemma_sv_multiseed")]
    discovered = _discover(roots)

    def selected(explicit: Sequence[Path], category: str) -> Sequence[Path]:
        return explicit if explicit else discovered[category]

    try:
        report = summarize_paths(
            selected(args.recovery, "recovery"),
            selected(args.control, "control"),
            certificate_paths=selected(args.certificate, "certificate"),
            audit_paths=selected(args.audit, "audit"),
            cross_corpus_paths=selected(args.cross_corpus, "cross_corpus"),
            zero_shot_paths=selected(args.zero_shot, "zero_shot"),
        )
    except (AggregationError, OSError) as error:
        parser.error(str(error))

    rendered = deterministic_json(report)
    if args.out == "-":
        print(rendered, end="")
    else:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

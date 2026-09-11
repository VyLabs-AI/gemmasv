"""Aggregate corrected, stream-matched 4B recovery/control seed pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from gemma_sv.recovery_protocol import validate_stage2_pair, write_json_atomic
from gemma_sv.summarize_multiseed import student_t_ci95


SCHEMA = "gemma-sv-4b-multiseed-v1"
MODEL = "google/gemma-3-4b-pt"
MODEL_REVISION = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"


class ScalingSummaryError(RuntimeError):
    """Raised when a purported scaling pair is incomplete or mismatched."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ScalingSummaryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScalingSummaryError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScalingSummaryError(f"{label} must be numeric")
    return float(value)


def _validate_config(
    payload: dict[str, Any],
    *,
    seed: int,
    mode: str,
    expected_steps: int | None,
) -> None:
    if payload.get("schema") != "gemma-sv-recovery-v2":
        raise ScalingSummaryError(f"seed {seed} {mode}: wrong schema")
    if payload.get("mode") != mode:
        raise ScalingSummaryError(f"seed {seed}: expected mode {mode!r}")
    if payload.get("model") != MODEL:
        raise ScalingSummaryError(f"seed {seed} {mode}: wrong model")
    config = payload.get("config") or {}
    if config.get("model_revision") != MODEL_REVISION:
        raise ScalingSummaryError(f"seed {seed} {mode}: wrong model revision")
    for name in ("seed", "data_seed", "init_seed"):
        if int(config.get(name, -1)) != seed:
            raise ScalingSummaryError(f"seed {seed} {mode}: {name} mismatch")
    if int(config.get("rank", -1)) != 8:
        raise ScalingSummaryError(f"seed {seed} {mode}: rank must be 8")
    if expected_steps is not None and int(config.get("stage2_steps", -1)) != expected_steps:
        raise ScalingSummaryError(f"seed {seed} {mode}: stage-2 steps differ")
    expected_offset = int(config.get("stage1_steps", -1)) + 1
    if int(payload.get("stage2_offset_batches", -1)) != expected_offset:
        raise ScalingSummaryError(f"seed {seed} {mode}: stage-2 offset mismatch")
    stream = payload.get("stage2_stream") or {}
    if int(stream.get("batches", -1)) != int(config.get("stage2_steps", -2)):
        raise ScalingSummaryError(f"seed {seed} {mode}: incomplete stage-2 stream")
    if not stream.get("sha256"):
        raise ScalingSummaryError(f"seed {seed} {mode}: missing stream fingerprint")


def summarize(root: Path, seeds: Iterable[int]) -> dict[str, Any]:
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) < 3:
        raise ScalingSummaryError("at least three corrected seed pairs are required")
    if len(set(seed_list)) != len(seed_list):
        raise ScalingSummaryError("seed list contains duplicates")

    rows: list[dict[str, Any]] = []
    expected_steps: int | None = None
    eval_hash: str | None = None
    seen_streams: set[str] = set()

    for seed in seed_list:
        recovered_path = root / f"seed-{seed}" / "recovered" / "results.json"
        control_path = root / f"seed-{seed}" / "control" / "control_results.json"
        recovered = _read(recovered_path)
        control = _read(control_path)
        config = recovered.get("config") or {}
        if expected_steps is None:
            expected_steps = int(config.get("stage2_steps", -1))
        _validate_config(
            recovered,
            seed=seed,
            mode="recovered",
            expected_steps=expected_steps,
        )
        _validate_config(
            control,
            seed=seed,
            mode="control",
            expected_steps=expected_steps,
        )
        try:
            validate_stage2_pair(recovered, control)
        except ValueError as exc:
            raise ScalingSummaryError(f"seed {seed}: {exc}") from exc
        if not (control.get("pairing") or {}).get("valid"):
            raise ScalingSummaryError(f"seed {seed}: control pairing is not valid")

        current_eval_hash = str((recovered.get("eval") or {}).get("token_ids_sha256", ""))
        if not current_eval_hash:
            raise ScalingSummaryError(f"seed {seed}: missing evaluation hash")
        if eval_hash is None:
            eval_hash = current_eval_hash
        elif current_eval_hash != eval_hash:
            raise ScalingSummaryError("evaluation token hashes differ across seeds")

        stream_hash = str((recovered.get("stage2_stream") or {})["sha256"])
        if stream_hash in seen_streams:
            raise ScalingSummaryError("different seeds unexpectedly share a stage-2 stream")
        seen_streams.add(stream_hash)

        recovered_ppl = _number(recovered.get("ppl_final"), "recovered perplexity")
        control_ppl = _number(control.get("ppl_control"), "control perplexity")
        original_ppl = _number(recovered.get("ppl_orig"), "original perplexity")
        graft_untrained_ppl = _number(
            recovered.get("ppl_graft0"),
            "untrained graft perplexity",
        )
        rows.append(
            {
                "seed": seed,
                "original_ppl": original_ppl,
                "graft_untrained_ppl": graft_untrained_ppl,
                "recovered_ppl": recovered_ppl,
                "control_ppl": control_ppl,
                "utility_cost_percent": 100.0 * (recovered_ppl / control_ppl - 1.0),
                "stage2_minutes": _number(
                    recovered.get("stage2_minutes"),
                    "stage-2 minutes",
                ),
                "stage2_stream_sha256": stream_hash,
                "recovery_adapter_sha256": recovered.get("adapter_sha256"),
                "control_adapter_sha256": control.get("adapter_sha256"),
                "recovery_report_sha256": _sha256(recovered_path),
                "control_report_sha256": _sha256(control_path),
            }
        )

    rows.sort(key=lambda row: int(row["seed"]))
    return {
        "schema": SCHEMA,
        "status": "complete",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "n_seeds": len(rows),
        "seeds": [row["seed"] for row in rows],
        "stage2_steps": expected_steps,
        "evaluation_token_ids_sha256": eval_hash,
        "seed_rows": rows,
        "original_ppl": student_t_ci95([row["original_ppl"] for row in rows]),
        "graft_untrained_ppl": student_t_ci95(
            [row["graft_untrained_ppl"] for row in rows]
        ),
        "recovered_ppl": student_t_ci95([row["recovered_ppl"] for row in rows]),
        "control_ppl": student_t_ci95([row["control_ppl"] for row in rows]),
        "utility_cost_percent": student_t_ci95(
            [row["utility_cost_percent"] for row in rows]
        ),
        "inference_unit": "independent recovery/control training seed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/gemma_sv_4b_multiseed"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/gemma_sv_4b_multiseed/summary.json"),
    )
    args = parser.parse_args()
    result = summarize(args.root, args.seeds)
    write_json_atomic(args.out, result)
    print(json.dumps(result["utility_cost_percent"], sort_keys=True))
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

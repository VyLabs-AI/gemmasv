"""Run and merge sharded whole-record Leak@k on the frozen 4B configuration."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from gemma_sv.eval_whole_record_unlearning import _summaries
from gemma_sv.recovery_protocol import write_json_atomic


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "benchmarks" / "boundary_attack_suite_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_report(
    path: Path,
    condition: str,
    *,
    samples: int,
    k_values: list[int],
) -> dict:
    report = json.loads(path.read_text())
    rows = report["whole_record"]["records"]
    if (
        len(rows) != 1
        or set(rows[0]["conditions"]) != {condition}
        or int(report["sampling"]["samples"]) != samples
        or list(report["sampling"]["k"]) != k_values
    ):
        raise ValueError(f"incomplete attack shard: {path}")
    return report


def _provenance_signature(report: dict) -> dict:
    provenance = copy.deepcopy(report["provenance"])
    geometry = provenance["geometry"]
    geometry.pop("record_start", None)
    geometry.pop("records", None)
    sampling = report["sampling"]
    return {
        "model": report["model"],
        "model_revision": report["model_revision"],
        "lora": report["lora"],
        "nu": report["nu"],
        "preserve_prefix_mass": report["preserve_prefix_mass"],
        "per_boundary_box": report["per_boundary_box"],
        "solver_seed": report["solver_seed"],
        "provenance": provenance,
        "sampling": {
            key: sampling[key]
            for key in (
                "temperature",
                "top_p",
                "max_new_tokens",
                "seed",
                "kv_cache",
            )
        },
    }


def merge_shards(config: dict, output_root: Path, output: Path, *, smoke: bool) -> None:
    sampling = dict(config["whole_record_sampling"])
    conditions = list(sampling["conditions"])
    indices = list(config["admitted_record_indices"])
    if smoke:
        indices = indices[:1]
        sampling["samples_per_prompt"] = 2
        sampling["k"] = [1, 2]

    template = None
    signature = None
    combined_rows = []
    sources = {}
    for index in indices:
        merged = None
        for condition in conditions:
            shard = output_root / f"record-{index:03d}-{condition}.json"
            report = _load_report(
                shard,
                condition,
                samples=sampling["samples_per_prompt"],
                k_values=sampling["k"],
            )
            current_signature = _provenance_signature(report)
            if signature is None:
                signature = current_signature
            elif current_signature != signature:
                raise ValueError("attack shard provenance mismatch")
            sources[shard.name] = _sha256(shard)
            row = report["whole_record"]["records"][0]
            if int(row["manifest_index"]) != index:
                raise ValueError("attack shard manifest index mismatch")
            if merged is None:
                template = template or report
                merged = {
                    key: copy.deepcopy(value)
                    for key, value in row.items()
                    if key != "conditions"
                }
                merged["conditions"] = {}
            if row["record_id"] != merged["record_id"]:
                raise ValueError("record identity drift across attack shards")
            merged["conditions"][condition] = copy.deepcopy(
                row["conditions"][condition]
            )
        combined_rows.append(merged)

    exact, composite = _summaries(
        combined_rows,
        conditions,
        sampling["k"],
    )
    merged_report = copy.deepcopy(template)
    merged_report["evaluation"] = "frozen 4B whole-record behavioral attacks"
    merged_report["attack_suite"] = {
        "name": config["name"],
        "version": config["version"],
        "selected_record_indices": indices,
        "selection": "all six records admitted by the frozen confirmation",
        "smoke": smoke,
        "shard_sha256": sources,
    }
    merged_report["sampling"]["samples"] = sampling["samples_per_prompt"]
    merged_report["sampling"]["k"] = sampling["k"]
    merged_report["sampling"]["conditions"] = conditions
    merged_report["provenance"]["geometry"]["record_start"] = None
    merged_report["provenance"]["geometry"]["records"] = len(indices)
    merged_report["whole_record"] = {
        "manifest": template["whole_record"]["manifest"],
        "manifest_version": template["whole_record"]["manifest_version"],
        "mode": "behavioral",
        "deletion_scope": "record",
        "attempted": len(indices),
        "admitted": len(combined_rows),
        "admission_rate": 1.0,
        "admission_gates": template["whole_record"]["admission_gates"],
        "exact_phrase_leak_at_k": exact,
        "composite_any_leak_at_k": composite,
        "records": combined_rows,
        "rejected_records": [],
    }
    write_json_atomic(output, merged_report)
    print(f"wrote {output}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output-root",
        default="outputs/gemma_sv_boundary_attack_suite/shards",
    )
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_boundary_attack_suite/whole_record.json",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sampling = dict(config["whole_record_sampling"])
    indices = list(config["admitted_record_indices"])
    if args.smoke:
        indices = indices[:1]
        sampling["samples_per_prompt"] = 2
        sampling["k"] = [1, 2]

    if not args.merge_only:
        model = config["model"]
        for index in indices:
            for condition in sampling["conditions"]:
                destination = (
                    output_root / f"record-{index:03d}-{condition}.json"
                )
                if destination.exists():
                    try:
                        _load_report(
                            destination,
                            condition,
                            samples=sampling["samples_per_prompt"],
                            k_values=sampling["k"],
                        )
                    except (KeyError, ValueError, json.JSONDecodeError):
                        pass
                    else:
                        print(f"complete; skipping {destination}")
                        continue
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "gemma_sv.eval_whole_record_unlearning",
                    "--manifest",
                    str(ROOT / "benchmarks" / config["manifest"]),
                    "--record-start",
                    str(index),
                    "--records",
                    "1",
                    "--conditions",
                    condition,
                    "--samples",
                    str(sampling["samples_per_prompt"]),
                    "--batch-size",
                    str(sampling["batch_size"]),
                    "--max-new-tokens",
                    str(sampling["max_new_tokens"]),
                    "--temperature",
                    str(sampling["temperature"]),
                    "--top-p",
                    str(sampling["top_p"]),
                    "--k",
                    ",".join(str(value) for value in sampling["k"]),
                    "--model",
                    model["id"],
                    "--model-revision",
                    model["revision"],
                    "--dataset-revision",
                    config["dataset_revision"],
                    "--lora",
                    "none",
                    "--device",
                    "mps",
                    "--window",
                    "1024",
                    "--n-fill",
                    "22",
                    "--prefix-fillers",
                    "11",
                    "--seed",
                    str(sampling["seed"]),
                    "--nu",
                    "0.7",
                    "--solver-seed",
                    "0",
                    "--preserve-prefix-mass",
                    "--per-boundary-box",
                    "--kv-cache",
                    "--out",
                    str(destination),
                ]
                subprocess.run(command, check=True)

    merge_shards(
        config,
        output_root,
        Path(args.out),
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

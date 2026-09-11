"""Run the separately specified frozen-4B journal cost/utility bridge, offline.

Run from the repository root with ``python -m journal_studies.gemmasv.run_comparison``.
No decoded generation is performed. See protocol.json for inference limits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

# These are requirements of this standalone study, not changes to old studies.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def paired_ratio_summary(pairs, *, expected_records=8, samples=10000, seed=0):
    """Each pair is one record's numerator/denominator median latency."""
    pairs = list(pairs)
    if any(not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0 for a, b in pairs):
        raise ValueError("latencies must be finite and positive")
    result = {"unit": "record", "paired_records": len(pairs), "expected_records": expected_records,
              "complete_cohort": len(pairs) == expected_records, "ratio_direction": "numerator / denominator; >1 means numerator slower"}
    if not pairs:
        return result
    logs = np.log(np.asarray([a / b for a, b in pairs], dtype=np.float64))
    result["geometric_mean_ratio"] = float(np.exp(logs.mean()))
    result["record_ratios"] = np.exp(logs).tolist()
    if len(pairs) >= 2:
        rng = np.random.default_rng(seed)
        boot = np.exp(logs[rng.integers(0, len(logs), size=(samples, len(logs)))].mean(axis=1))
        result["bootstrap_95_percentile_interval"] = np.quantile(boot, [0.025, 0.975]).tolist()
        result["bootstrap_samples"] = samples
    else:
        result["bootstrap_95_percentile_interval"] = None
    return result


def literal_delete(token_ids, positions):
    positions = tuple(sorted(set(int(x) for x in positions)))
    if not positions or positions[0] < 0 or positions[-1] >= len(token_ids):
        raise ValueError("deletion positions must be nonempty and in range")
    removed = set(positions)
    return tuple(int(token) for index, token in enumerate(token_ids) if index not in removed)


def compact_failure(exc):
    # Exception messages can contain dataset text; never serialize them here.
    return {"status": "failed", "error_type": type(exc).__name__, "error": "omitted_in_compact_report"}


def _git(*args):
    completed = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return completed.stdout if completed.returncode == 0 else None


def capture_provenance(output, protocol, args):
    """Must finish before ensure_loaded, prefill, or scoring."""
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source_snapshot"
    source.mkdir()
    paths = set()
    for folder in ("gemma_sv", "svattn", "cp_svm", "journal_studies/gemmasv"):
        paths.update((ROOT / folder).rglob("*.py"))
    paths.add(PROTOCOL_PATH)
    paths.add(ROOT / protocol["manifest"])
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = sha256(destination)
    versions = {}
    for package in ("torch", "transformers", "mlx", "datasets", "numpy", "scipy", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    snapshot = Path(args.model_snapshot).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError("local model snapshot is missing")
    model_files = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file():
            entry = {"bytes": path.stat().st_size, "resolved_path": str(path.resolve())}
            if path.suffix == ".json" or path.name == "tokenizer.model":
                entry["sha256"] = sha256(path)
            else:
                entry["weight_content_hash_verified_this_run"] = False
            model_files[path.name] = entry
    if "config.json" not in model_files or not any(name.endswith(".safetensors") for name in model_files):
        raise FileNotFoundError("local model snapshot lacks config or weights")
    tracked_diff = _git("diff", "HEAD", "--", "gemma_sv", "svattn", "cp_svm", "journal_studies/gemmasv")
    provenance = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL_PATH), "manifest_sha256": sha256(ROOT / protocol["manifest"]),
        "source_files_sha256": hashes, "source_snapshot": "source_snapshot",
        "git_commit": (_git("rev-parse", "HEAD") or "").strip(),
        "git_status": _git("status", "--porcelain", "--untracked-files=normal"),
        "tracked_diff_sha256": hashlib.sha256((tracked_diff or "").encode()).hexdigest(),
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "model_snapshot": str(snapshot), "model_files": model_files,
        "model_weights_rehashed": False,
        "offline_required": True, "execution_arguments": vars(args),
        "source_text_in_result": False,
    }
    atomic_write(output / "protocol.json", protocol)
    atomic_write(output / "provenance.json", provenance)
    return provenance


def build_record(runtime, spec, forget, retain, fillers, config):
    from gemma_sv.eval_whole_record_unlearning import _record_memory
    from gemma_sv.eval_persistent_deletion_baselines import Probe, _stem_prompt, _target_ids
    from gemma_sv.robust_eval import extract_secret
    from gemma_sv.persistent_deletion import token_ids_digest

    deleted_rows = [forget[int(index)] for index in spec["forget_indices"]]
    retained_row = retain[int(spec["retain_index"])]
    present, _, _ = _record_memory(runtime, deleted_rows, retained_row, fillers,
        window=config["window"], n_fill=config["n_fill"], prefix_fillers=config["prefix_fillers"])
    original = tuple(int(x) for x in present.token_ids)
    positions = tuple(int(x) for x in present.positions["forget"])
    edited = literal_delete(original, positions)
    probes = []
    for index, row in enumerate([*deleted_rows, retained_row]):
        span = extract_secret(str(row["question"]), str(row["answer"]))
        if span is None:
            raise ValueError("fixed cohort record lacks an established secret span")
        prompt = _stem_prompt(str(row["question"]), span.stem)
        probes.append(Probe(probe_id=f"deleted_field_{index}" if index < 3 else "retained_field",
            kind="deleted" if index < 3 else "retained", prompt=prompt, target=span.secret,
            target_ids=_target_ids(runtime, prompt, span.secret), field_index=index if index < 3 else None))
    metadata = {"original_tokens": len(original), "edited_tokens": len(edited), "deleted_tokens": len(positions),
        "original_token_ids_sha256": token_ids_digest(original), "edited_token_ids_sha256": token_ids_digest(edited),
        "literal_edit_preserves_retained_tokens": True,
        "probes": [{"probe_id": p.probe_id, "kind": p.kind, "target_tokens": len(p.target_ids),
            "target_token_ids_sha256": token_ids_digest(p.target_ids),
            "prompt_token_ids_sha256": token_ids_digest(runtime.tokenizer(p.prompt, add_special_tokens=False).input_ids)} for p in probes]}
    return original, edited, positions, probes, metadata


def score_comparison(probes, scores, present_scores, reference_scores):
    from gemma_sv.eval_persistent_deletion_baselines import _behavioral_metrics
    return _behavioral_metrics(probes, scores, present_scores, reference_scores, compact=True)


def cross_architecture(probes, scores, base_scores):
    from gemma_sv.persistent_deletion import full_vocabulary_kl
    return {"reference": "ungrafted literal full repack", "is_deletion_certificate": False,
        "probes": [{"probe_id": p.probe_id,
            "mean_target_log_probability_graft_minus_base_nats": float(scores[p.probe_id]["mean_log_probability"] - base_scores[p.probe_id]["mean_log_probability"]),
            "first_token_kl_base_repack_to_graft_nats": full_vocabulary_kl(base_scores[p.probe_id]["first_log_probs"], scores[p.probe_id]["first_log_probs"])} for p in probes]}


def summarize(report):
    expected = len(report["selected_record_ids"])
    by_id = {row["record_id"]: row for row in report["records"]}
    comparisons = [("graft_masked_refit_proxy", "graft_full_repack"),
                   ("graft_cache_delete_shift", "graft_full_repack"),
                   ("graft_masked_refit_proxy", "base_full_repack"),
                   ("graft_full_repack", "base_full_repack")]
    ratios = {}
    for numerator, denominator in comparisons:
        for metric in ("update_seconds", "query_seconds", "end_to_end_seconds"):
            pairs = []
            for record in by_id.values():
                arms = record.get("arms", {})
                a, b = arms.get(numerator, {}), arms.get(denominator, {})
                if (a.get("status") == b.get("status") == "completed"
                        and "metrics" in a and "metrics" in b):
                    pairs.append((a["timing"][metric]["median"], b["timing"][metric]["median"]))
            ratios[f"{numerator}_over_{denominator}:{metric}"] = paired_ratio_summary(pairs, expected_records=expected)
    arms_summary = {}
    for name in report["protocol"]["arms"]:
        values = [r.get("arms", {}).get(name, {}) for r in by_id.values()]
        completed = [v for v in values if v.get("status") == "completed" and "metrics" in v]
        arms_summary[name] = {"expected_records": expected, "attempted": sum(bool(v) for v in values),
            "completed": len(completed), "failed": sum(v.get("status") == "failed" for v in values),
            "measurement_without_valid_reference": sum(v.get("status") == "completed" and "metrics" not in v for v in values),
            "boundary_precheck_full_repack_fallback_records": sum(v.get("full_repack_fallback", False) for v in completed)}
        if completed:
            arms_summary[name].update({"mean_deleted_target_log_probability": float(np.mean([v["metrics"]["deleted_target_quality"]["mean_log_probability"] for v in completed])),
                "mean_retained_target_log_probability": float(np.mean([v["metrics"]["retained_quality"]["score"]["mean_log_probability"] for v in completed])),
                "mean_deleted_probe_kl_to_own_repack_nats": float(np.mean([v["metrics"]["behavioral_kl_to_repack"]["deleted_probe_mean_nats"] for v in completed])),
                "mean_retained_probe_kl_to_own_repack_nats": float(np.mean([v["metrics"]["behavioral_kl_to_repack"]["retained_probe_nats"] for v in completed]))})
    return {"smoke_only": report["smoke"], "all_record_denominator": expected,
            "arms": arms_summary, "paired_latency_ratios": ratios,
            "interpretation": "Teacher-forced target scores; not decoded leakage. No exact-decrement latency measured. Incomplete-arm aggregates show their denominators and are descriptive only."}


def run(args):
    from datasets import load_dataset
    from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
    from gemma_sv.eval_persistent_deletion_baselines import (QueryState, _benchmark_existing_state,
        _benchmark_method, _score_probe_suite, _release_runtime, _seed_everything, _synchronize)
    from gemma_sv.persistent_deletion import cache_delete_and_shift

    protocol = json.loads(PROTOCOL_PATH.read_text())
    cfg = protocol["configuration"]
    output = Path(args.out).expanduser().resolve()
    provenance = capture_provenance(output, protocol, args)
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    records = manifest["records"][:1] if args.smoke else manifest["records"]
    if len(manifest["records"]) != 8:
        raise ValueError("frozen cohort must contain eight records")
    warmup, repeats = (0, 1) if args.smoke else (1, 3)
    report = {"schema": protocol["schema"], "status": "loading_local_data", "smoke": args.smoke,
        "protocol": protocol, "protocol_sha256": provenance["protocol_sha256"],
        "selected_record_ids": [str(r["record_id"]) for r in records],
        "execution": {"warmup": warmup, "repeats": repeats, "device": args.device}, "records": []}
    result_path = output / "results.json"
    atomic_write(result_path, report)
    try:
        # With HF_*_OFFLINE set before imports, this uses existing local cache only.
        loaded = {}
        for name in ("forget10", "retain90"):
            dataset = load_dataset(protocol["dataset"], name, split="train", revision=protocol["dataset_revision"])
            cache_paths = [Path(entry["filename"]) for entry in dataset.cache_files]
            # Offline datasets may choose the newest cache while ignoring revision.
            # Accept only the directory bound to this protocol's exact revision.
            if not cache_paths or any(path.parent.name != protocol["dataset_revision"] for path in cache_paths):
                raise RuntimeError("offline dataset cache revision does not match protocol")
            provenance.setdefault("dataset_cache", {})[name] = {
                "fingerprint": dataset._fingerprint,
                "files": [{"path": str(path), "sha256": sha256(path)} for path in cache_paths],
            }
            loaded[name] = list(dataset)
        forget, retain = loaded["forget10"], loaded["retain90"]
        fillers = [str(row["answer"]) for row in retain[:max(cfg["n_fill"] + 8, 32)]]
        # Bind loaded local dataset content without serializing any source text.
        for name, rows in (("forget10", forget), ("retain90", retain)):
            provenance.setdefault("loaded_dataset_content_sha256", {})[name] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        atomic_write(output / "provenance.json", provenance)
    except Exception as exc:
        report.update(status="failed_loading_local_data", failure=compact_failure(exc))
        atomic_write(result_path, report)
        raise
    base_references = {}
    for architecture in ("base", "graft"):
        runtime = None
        try:
            _seed_everything(0)
            config = RuntimeConfig(model_id=provenance["model_snapshot"], model_revision=protocol["model_revision"],
                lora_path=None, device=args.device, dtype="float32", generation_tokens=1,
                window=cfg["window"], graft_enabled=architecture == "graft", nu=cfg["nu"],
                preserve_prefix_mass=architecture == "graft", per_boundary_box=architecture == "graft", solver_seed=0)
            runtime = GemmaRuntime(config)
            report["status"] = f"loading_{architecture}"
            atomic_write(result_path, report)
            runtime.ensure_loaded()
            report.setdefault("runtime", {})[architecture] = {"config": asdict(config), "load_seconds": runtime.load_seconds}
            for index, spec in enumerate(records):
                print(f"{architecture} record {index + 1}/{len(records)} {spec['record_id']}", flush=True)
                _seed_everything(index)
                row = next((r for r in report["records"] if r["record_id"] == str(spec["record_id"])), None)
                if row is None:
                    row = {"record_id": str(spec["record_id"]), "manifest_index": index, "arms": {}, "excluded_by_admission": False}
                    report["records"].append(row)
                try:
                    original, edited, positions, probes, metadata = build_record(runtime, spec, forget, retain, fillers, cfg)
                    if "context" in row and row["context"] != metadata:
                        raise RuntimeError("architectures produced unmatched contexts or probes")
                    row["context"] = metadata
                    _synchronize(args.device)
                    start = time.perf_counter()
                    memory = runtime.prefill_persistent(list(original))
                    _synchronize(args.device)
                    row.setdefault("shared_original_prefill", {})[architecture] = {"seconds": time.perf_counter() - start}
                    query = lambda state: _score_probe_suite(runtime, state, probes)
                    present_scores, present_timing = _benchmark_existing_state(state=QueryState(memory), query=query,
                        device=args.device, warmup=warmup, repeats=repeats)
                    method_scores, measured = {}, {}
                    builders = {"full_repack": lambda: QueryState(runtime.prefill_persistent(list(edited)))}
                    if architecture == "graft":
                        builders["masked_refit_proxy"] = lambda: QueryState(runtime.delete_persistent(memory, positions, kind="fp32_masked_refit_proxy"))
                        def cache_build():
                            state, diagnostics = cache_delete_and_shift(memory, positions, edited_token_ids=edited)
                            return QueryState(state, update_diagnostics=diagnostics)
                        builders["cache_delete_shift"] = cache_build
                    order = list(builders)
                    if architecture == "graft":
                        shift = index % len(order)
                        order = order[shift:] + order[:shift]
                    row.setdefault("method_order", {})[architecture] = order
                    for name in order:
                        try:
                            state, scores, timing = _benchmark_method(build=builders[name], query=query,
                                device=args.device, warmup=warmup, repeats=repeats, operation_seed=index)
                            method_scores[name] = scores
                            measured[name] = {"status": "completed", "timing": timing,
                                "deletion_kind": state.memory.deletion_kind,
                                "full_repack_fallback": state.memory.fallback_reason is not None,
                                "fallback_reason": state.memory.fallback_reason,
                                "diagnostic_only": name == "cache_delete_shift"}
                            del state
                        except Exception as exc:
                            measured[name] = compact_failure(exc)
                        row["arms"][f"{architecture}_{name}"] = measured[name]
                        report["status"] = f"running_{architecture}"
                        atomic_write(result_path, report)
                    reference = method_scores.get("full_repack")
                    if reference is None:
                        raise RuntimeError("fresh reference arm failed")
                    method_scores["present"] = present_scores
                    measured["present"] = {"status": "completed", "timing": present_timing, "full_repack_fallback": False}
                    for name, scores in method_scores.items():
                        arm = measured[name]
                        arm["reference_architecture"] = architecture
                        arm["metrics"] = score_comparison(probes, scores, present_scores, reference)
                        if name == "full_repack":
                            self_kl = arm["metrics"]["behavioral_kl_to_repack"]
                            if max(self_kl["deleted_probe_max_nats"], self_kl["retained_probe_nats"]) > protocol["tolerances"]["self_reference_kl_nats"]:
                                raise RuntimeError("self-reference KL exceeds numerical tolerance")
                        if architecture == "graft" and str(spec["record_id"]) in base_references:
                            arm["cross_architecture"] = cross_architecture(probes, scores, base_references[str(spec["record_id"])])
                        row["arms"][f"{architecture}_{name}"] = arm
                    if architecture == "base":
                        base_references[str(spec["record_id"])] = reference
                    del memory, method_scores, measured, present_scores
                except Exception as exc:
                    row.setdefault("architecture_failures", {})[architecture] = compact_failure(exc)
                atomic_write(result_path, report)
        except Exception as exc:
            report.setdefault("architecture_failures", {})[architecture] = compact_failure(exc)
            atomic_write(result_path, report)
        finally:
            _release_runtime(runtime)
            runtime = None
    report["summary"] = summarize(report)
    complete = all(value["completed"] == len(records) for value in report["summary"]["arms"].values())
    report["status"] = "completed" if complete else "completed_with_failures"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write(result_path, report)
    print(f"{report['status']}: {result_path}", flush=True)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New output directory; existing directories are rejected")
    parser.add_argument("--model-snapshot", default=str(DEFAULT_SNAPSHOT), help="Existing local 4B PT snapshot only")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke", action="store_true", help="First record, zero warmup, one repeat; separate pilot only")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

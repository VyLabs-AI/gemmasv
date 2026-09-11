"""Float64 output certificates for synthetic or TOFU whole records.

The behavioral benchmark is fp32/MPS. This companion command rebuilds the same
packed memory in float64, seals it once, and uses the runtime's frozen
per-boundary boxes and bandwidths to compare decrement with retained-key refit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_whole_record_unlearning import (
    DEFAULT_MANIFEST,
    _adapter_sha256,
    _composite_prompt,
    _field_from_answer,
    _normalize_lora,
    _record_memory,
    _sha256,
    _synthetic_memory,
)
from gemma_sv.robust_eval import extract_secret


HERO_KL_TARGET = 1e-6
TOFU_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _kl(log_p: np.ndarray, log_q: np.ndarray) -> float:
    probability = np.exp(log_p)
    return max(0.0, float(np.sum(probability * (log_p - log_q))))


def certify_probe(
    runtime: GemmaRuntime,
    states: dict,
    overrides: dict,
    prompt: str,
    target: str,
) -> dict:
    started = time.perf_counter()
    target_ids = runtime.target_ids(target, prompt)
    distributions = {}
    probabilities = {}
    for case in ("exact", "refit", "decay"):
        score = runtime.score_persistent(
            states[case],
            prompt,
            target_ids,
        )
        distributions[case] = score["first_log_probs"]
        probabilities[case] = score["geometric_mean_probability"]
    return {
        "kl_exact_vs_refit_nats": _kl(
            distributions["exact"],
            distributions["refit"],
        ),
        "kl_decay_vs_refit_nats": _kl(
            distributions["decay"],
            distributions["refit"],
        ),
        "target_probabilities": probabilities,
        "top_tokens": {
            case: runtime.top_tokens(distribution)
            for case, distribution in distributions.items()
        },
        "fixed_c_feasible": overrides["fixed_c_feasible"],
        "decrement_fallbacks": overrides["n_fallback"],
        "head_gates": overrides["n_solves"],
        "max_functional_deviation": overrides[
            "max_functional_deviation"
        ],
        "max_candidate_deviation": overrides["max_candidate_deviation"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv=None) -> int:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--behavior-report",
        default="outputs/gemma_sv_eval/whole_record_synthetic_v1.json",
    )
    parser.add_argument("--record-id")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="certificate every manifest record, independent of recall admission",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision", default=TOFU_REVISION)
    parser.add_argument("--lora")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_eval/whole_record_certificates.json",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    behavior_path = Path(args.behavior_report)
    manifest = json.loads(manifest_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    provenance = behavior.get("provenance") or {}
    behavior_manifest = provenance.get("manifest") or {}
    behavior_model = provenance.get("model") or {}
    behavior_adapter = provenance.get("adapter") or {}
    behavior_dataset = provenance.get("dataset") or {}
    geometry = provenance.get("geometry") or {}
    if behavior_manifest.get("sha256") != _sha256(manifest_path):
        parser.error("behavior report manifest SHA-256 does not match --manifest")
    model = args.model or behavior_model.get("id")
    model_revision = args.model_revision or behavior_model.get("resolved_revision")
    lora = _normalize_lora(args.lora or behavior_adapter.get("path"))
    if not model or not model_revision:
        parser.error("behavior report must bind model id and resolved revision")
    if args.model and args.model != behavior_model.get("id"):
        parser.error("--model differs from behavior report")
    if args.model_revision and args.model_revision != behavior_model.get(
        "resolved_revision"
    ):
        parser.error("--model-revision differs from behavior report")
    adapter_hash = _adapter_sha256(lora)
    if adapter_hash != behavior_adapter.get("content_sha256"):
        parser.error("adapter SHA-256 differs from behavior report")
    try:
        window = int(geometry["window"])
        n_fill = int(geometry["n_fill"])
        prefix_fillers = int(geometry["prefix_fillers"])
        nu = float(geometry["nu"])
        per_boundary_box = bool(geometry["per_boundary_box"])
        preserve_prefix_mass = bool(geometry["preserve_prefix_mass"])
        solver_seed = int(geometry["solver_seed"])
    except (KeyError, TypeError, ValueError):
        parser.error("behavior report does not contain valid packing geometry")
    behavior_rows = list(behavior["whole_record"]["records"])
    if args.include_rejected:
        behavior_rows += list(behavior["whole_record"]["rejected_records"])
    admitted_ids = {row["record_id"] for row in behavior_rows}
    if args.record_id:
        admitted_ids &= {args.record_id}
    specs = [
        spec for spec in manifest["records"]
        if spec["record_id"] in admitted_ids
    ]
    if not specs:
        parser.error("no admitted record matches --record-id")

    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=model,
            lora_path=lora,
            device=args.device,
            dtype="float64",
            generation_tokens=1,
            window=window,
            model_revision=model_revision,
            nu=nu,
            preserve_prefix_mass=preserve_prefix_mass,
            per_boundary_box=per_boundary_box,
            solver_seed=solver_seed,
        )
    )
    runtime.ensure_loaded()
    forget_dataset = load_dataset(
        "locuslab/TOFU",
        "forget10",
        split="train",
        revision=args.dataset_revision,
    )
    retain_dataset = load_dataset(
        "locuslab/TOFU",
        "retain90",
        split="train",
        revision=args.dataset_revision,
    )
    forget_rows = list(forget_dataset)
    retain_rows = list(retain_dataset)
    fillers = [
        str(row["answer"])
        for row in retain_rows[: max(n_fill + prefix_fillers + 8, 32)]
    ]
    copies = int(manifest.get("selection_policy", {}).get("copies", 1))

    output = Path(args.out)
    report = {
        "evaluation": "whole-record float64 output certificates",
        "model": model,
        "model_revision": runtime.resolved_model_revision,
        "manifest": manifest["name"],
        "provenance": {
            "behavior_report_sha256": _sha256(behavior_path),
            "manifest_sha256": _sha256(manifest_path),
            "adapter_sha256": adapter_hash,
            "dataset_revision": args.dataset_revision,
            "dataset_fingerprints": {
                "forget10": behavior_dataset.get("forget_fingerprint"),
                "retain90": behavior_dataset.get("retain_fingerprint"),
            },
            "geometry": {
                "window": window,
                "n_fill": n_fill,
                "prefix_fillers": prefix_fillers,
                "nu": nu,
                "preserve_prefix_mass": preserve_prefix_mass,
                "per_boundary_box": per_boundary_box,
                "solver_seed": solver_seed,
            },
        },
        "record_count": 0,
        "hero_kl_target": HERO_KL_TARGET,
        "records": [],
    }
    for split, dataset in (
        ("forget10", forget_dataset),
        ("retain90", retain_dataset),
    ):
        expected = report["provenance"]["dataset_fingerprints"][split]
        actual = getattr(dataset, "_fingerprint", None)
        if expected is not None and actual != expected:
            parser.error(f"{split} fingerprint differs from behavior report")
    if args.resume and output.exists():
        existing = json.loads(output.read_text())
        existing_provenance = dict(existing.get("provenance") or {})
        existing_provenance.setdefault(
            "dataset_revision",
            report["provenance"]["dataset_revision"],
        )
        existing_provenance.setdefault(
            "dataset_fingerprints",
            report["provenance"]["dataset_fingerprints"],
        )
        if existing_provenance != report["provenance"]:
            parser.error("resume report provenance differs from current inputs")
        report["records"] = list(existing.get("records") or [])
        report["record_count"] = len(report["records"])
    completed_ids = {row["record_id"] for row in report["records"]}

    for spec in specs:
        if spec["record_id"] in completed_ids:
            print(f"  {spec['record_id']}: already complete", flush=True)
            continue
        if "question" in spec:
            present, _ = _synthetic_memory(
                runtime,
                spec,
                fillers,
                window=window,
                n_fill=n_fill,
                prefix_fillers=prefix_fillers,
                copies=copies,
            )
            fields = [
                _field_from_answer(spec["question"], spec["answer"], field)
                for field in spec["fields"]
            ]
        else:
            selected_forget = [
                forget_rows[int(index)] for index in spec["forget_indices"]
            ]
            selected_retain = retain_rows[int(spec["retain_index"])]
            present, _, _ = _record_memory(
                runtime,
                selected_forget,
                selected_retain,
                fillers,
                window=window,
                n_fill=n_fill,
                prefix_fillers=prefix_fillers,
            )
            fields = []
            for index, row in enumerate(selected_forget):
                question = str(row["question"])
                answer = str(row["answer"])
                span = extract_secret(question, answer)
                if span is None:
                    parser.error(
                        f"{spec['record_id']} field {index} has no secret span"
                    )
                fields.append(
                    {
                        "name": f"field_{index}",
                        "question": question,
                        "answer": answer,
                        "stem": span.stem,
                        "secret": span.secret,
                    }
                )
        probes = [
            (
                field["name"],
                f"\n\nQuestion: {field['question']}\nAnswer: {field['stem']}",
                field["secret"],
            )
            for field in fields
        ]
        probes.append(
            (
                "composite",
                _composite_prompt(fields),
                fields[0]["secret"],
            )
        )
        persistent = runtime.prefill_persistent(present.token_ids)
        overrides, states = runtime.persistent_certificate_states(
            persistent,
            present.positions["forget"],
        )
        results = {}
        for name, prompt, target in probes:
            print(f"  {spec['record_id']} / {name}", flush=True)
            results[name] = certify_probe(
                runtime,
                states,
                overrides,
                prompt,
                target,
            )
        max_kl = max(
            result["kl_exact_vs_refit_nats"]
            for result in results.values()
        )
        report["records"].append(
            {
                "record_id": spec["record_id"],
                "deleted_positions": len(present.positions["forget"]),
                "probe_certificates": results,
                "max_kl_exact_vs_refit_nats": max_kl,
                "hero_kl_target": HERO_KL_TARGET,
                "hero_kl_pass": max_kl <= HERO_KL_TARGET,
                "total_fallbacks": overrides["n_fallback"],
                "head_gates": overrides["n_head_gates"],
                "box_C_by_boundary": overrides["box_C_by_boundary"],
            }
        )
        report["record_count"] = len(report["records"])
        _write_json_atomic(output, report)
    _write_json_atomic(output, report)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

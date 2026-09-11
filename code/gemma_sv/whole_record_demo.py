"""One-command certified whole-record deletion demo.

The scenario is selected only from the predeclared benchmark's admitted set.
Every stored token belonging to the record is deleted atomically; all fields
are audited, an unrelated neighboring record is retained, and float64 output
certificates compare decrement with fixed-C refit-without.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from datasets import load_dataset

from gemma_sv.certify_whole_record import certify_probe
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_robust_unlearning import (
    DECAY,
    _condition_inputs,
    _stem_prompt,
    secret_probe_stats,
)
from gemma_sv.eval_whole_record_unlearning import (
    DEFAULT_MANIFEST,
    _composite_prompt,
    _field_from_answer,
    _synthetic_memory,
)


def _score_conditions(runtime, present, no_forget, field):
    prompt = _stem_prompt(field["question"], field["stem"])
    results = {}
    for condition in ("present", "decrement", "decay", "icul", "never"):
        memory, condition_prompt, request = _condition_inputs(
            present,
            no_forget,
            prompt,
            paired=False,
        )[condition]
        results[condition] = secret_probe_stats(
            runtime,
            memory,
            condition_prompt,
            field["secret"],
            request=request,
        )
    return prompt, results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--behavior-report",
        default="outputs/gemma_sv_eval/whole_record_synthetic_v1.json",
    )
    parser.add_argument(
        "--record-id",
        help=(
            "admitted record ID; default uses the declared maximin rule "
            "(largest minimum present first-token probability)"
        ),
    )
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--lora", default="outputs/gemma_sv_distill/lora_adapter")
    parser.add_argument("--fast-device", default="mps")
    parser.add_argument("--cert-device", default="cpu")
    parser.add_argument("--skip-certificate", action="store_true")
    parser.add_argument(
        "--certificate-report",
        help=(
            "reuse a recorded certificate for the same manifest/record; "
            "default searches outputs/gemma_sv_eval/whole_record_*_certificate.json"
        ),
    )
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_demo/whole_record_hero_results.json",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    behavior = json.loads(Path(args.behavior_report).read_text())
    admitted_rows = behavior["whole_record"]["records"]
    admitted = {row["record_id"] for row in admitted_rows}
    selection_rule = "explicit_record_id"
    if args.record_id is None:
        selected = max(
            admitted_rows,
            key=lambda row: min(
                field["secret_probe"]["first_token_probability"]
                for field in row["conditions"]["present"]["fields"]
            ),
        )
        args.record_id = selected["record_id"]
        selection_rule = "maximin_present_first_token_probability"
    if args.record_id not in admitted:
        parser.error(
            f"{args.record_id!r} is not in the predeclared admitted set"
        )
    spec = next(
        row for row in manifest["records"]
        if row["record_id"] == args.record_id
    )
    copies = int(manifest["selection_policy"].get("copies", 1))
    retain_rows = list(load_dataset("locuslab/TOFU", "retain90", split="train"))
    fillers = [str(row["answer"]) for row in retain_rows[:32]]

    fast = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora or None,
            device=args.fast_device,
            dtype="float32",
            generation_tokens=32,
            window=512,
        )
    )
    fast.ensure_loaded()
    present, no_forget = _synthetic_memory(
        fast,
        spec,
        fillers,
        window=512,
        n_fill=22,
        prefix_fillers=8,
        copies=copies,
    )
    fields = [
        _field_from_answer(spec["question"], spec["answer"], field)
        for field in spec["fields"]
    ]
    started = time.perf_counter()
    field_results = []
    print(
        f"WHOLE RECORD: {spec['record_id']} | {len(fields)} fields | "
        f"{len(present.positions['forget'])} token positions",
        flush=True,
    )
    for field in fields:
        prompt, conditions = _score_conditions(
            fast,
            present,
            no_forget,
            field,
        )
        field_results.append(
            {
                "name": field["name"],
                "value": field["secret"],
                "prompt": prompt,
                "conditions": conditions,
            }
        )
        print(
            f"  {field['name']}: present p1="
            f"{conditions['present']['first_token_probability']:.4f} -> "
            f"deleted {conditions['decrement']['first_token_probability']:.4f} "
            f"(never {conditions['never']['first_token_probability']:.4f}; "
            f"ICUL {conditions['icul']['first_token_probability']:.4f})",
            flush=True,
        )

    retain = _field_from_answer(
        spec["retain_question"],
        spec["retain_answer"],
        spec["retain_field"],
    )
    retain_prompt = _stem_prompt(retain["question"], retain["stem"])
    retain_present = secret_probe_stats(
        fast,
        present.token_ids,
        retain_prompt,
        retain["secret"],
    )
    retain_deleted = secret_probe_stats(
        fast,
        present.token_ids,
        retain_prompt,
        retain["secret"],
        request=GateRequest(drop_pos=present.positions["forget"]),
    )
    print(
        f"  NEIGHBOR RETAINED: {retain['name']} p1="
        f"{retain_present['first_token_probability']:.4f} -> "
        f"{retain_deleted['first_token_probability']:.4f}",
        flush=True,
    )

    certificates = {}
    certificate_source = None
    certificate_reports = (
        [Path(args.certificate_report)]
        if args.certificate_report
        else sorted(
            Path("outputs/gemma_sv_eval").glob(
                "whole_record_*_certificate.json"
            )
        )
    )
    matched = None
    matched_path = None
    if not args.skip_certificate:
        for certificate_report in certificate_reports:
            if not certificate_report.exists():
                continue
            recorded = json.loads(certificate_report.read_text())
            matched = next(
                (
                    row for row in recorded["records"]
                    if row["record_id"] == spec["record_id"]
                ),
                None,
            )
            if matched is not None:
                matched_path = certificate_report
                break
    if not args.skip_certificate and matched is not None:
        certificates = matched["probe_certificates"]
        certificate_source = f"recorded_float64_certificate:{matched_path}"
    elif not args.skip_certificate:
        certificate = GemmaRuntime(
            RuntimeConfig(
                model_id=args.model,
                lora_path=args.lora or None,
                device=args.cert_device,
                dtype="float64",
                generation_tokens=1,
                window=512,
            )
        )
        certificate.ensure_loaded()
        cert_memory, _ = _synthetic_memory(
            certificate,
            spec,
            fillers,
            window=512,
            n_fill=22,
            prefix_fillers=8,
            copies=copies,
        )
        for field in fields:
            prompt = _stem_prompt(field["question"], field["stem"])
            print(f"  CERTIFY {field['name']} ...", flush=True)
            certificates[field["name"]] = certify_probe(
                certificate,
                cert_memory.token_ids,
                cert_memory.positions["forget"],
                prompt,
                field["secret"],
            )
        print("  CERTIFY composite extraction ...", flush=True)
        certificates["composite"] = certify_probe(
            certificate,
            cert_memory.token_ids,
            cert_memory.positions["forget"],
            _composite_prompt(fields),
            fields[0]["secret"],
        )
        certificate_source = "live_float64_certificate"

    result = {
        "evidence": "live_behavior_with_certificate",
        "model": args.model,
        "manifest": manifest["name"],
        "record_id": spec["record_id"],
        "hero_selection_rule": selection_rule,
        "deletion_scope": "record",
        "memory_tokens": len(present.token_ids),
        "record_copies": copies,
        "deleted_token_positions": len(present.positions["forget"]),
        "fields": field_results,
        "neighbor": {
            "name": retain["name"],
            "value": retain["secret"],
            "present": retain_present,
            "after_record_deletion": retain_deleted,
        },
        "certificates": certificates,
        "certificate_source": certificate_source,
        "max_certificate_kl": (
            max(
                item["kl_exact_vs_refit_nats"]
                for item in certificates.values()
            )
            if certificates
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"artifacts -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build demo_site/assets/replay.json from recorded hosted benchmark runs.

Inputs are the outputs of ``gemma_sv.benchmark_hosted_demo`` (one per preset),
which contain the live float64 certificate and the labeled twin distributions.
Only preset scenarios are published; visitor text never reaches this file.

Run:
    .venv311/bin/python -m gemma_sv.make_replay_profile \
      --med outputs/gemma_sv_demo/hosted_benchmark_4b_med_field.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from gemma_sv.demo_server.contract import classify_admission, classify_certificate
from gemma_sv.demo_server.engine import preset_payloads
from gemma_sv.demo_server.scenarios import PRESETS


def build_conversation_log(preset, benchmark: dict) -> tuple[list[dict], int]:
    """Rebuild the exact packed conversation log using only the tokenizer.

    ``pack_memory`` is a pure function of the tokenizer, the selection, and the
    filler list, so this reproduces the live engine's log without model weights.
    """

    from transformers import AutoTokenizer

    from gemma_sv.demo_server.gemma_engine import (
        _QA_FILLERS,
        conversation_log_payload,
        pack_memory,
    )
    from gemma_sv.demo_server.span import SelectedSpan

    runtime = benchmark["runtime"]
    model_id = runtime["model_id"]
    window = int(runtime["window"])
    copies = int(runtime["memory_copies"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    selection = SelectedSpan(
        preset.memory_text, preset.secret_start, preset.secret_end
    )
    ids, _, segments = pack_memory(
        tokenizer,
        selection,
        list(_QA_FILLERS),
        with_fact=True,
        copies=copies,
        window=window,
    )
    return conversation_log_payload(segments, len(ids), window), len(ids)


def build_run(benchmark: dict) -> dict:
    steps = benchmark["steps"]
    ingest = steps["ingest"]["result"]
    recall = steps["recall"]["result"]
    forget = steps["forget"]["result"]
    icul = steps["attack_icul"]["result"]
    exact = steps["attack_exact"]["result"]
    certificate = steps["certificate"]["result"]
    twins = benchmark["twin_labeled"]
    if not twins or not twins.get("exact") or not twins.get("refit"):
        raise ValueError("benchmark run is missing labeled twin distributions")
    return {
        "source_scenario": benchmark["scenario"],
        "memory_tokens": ingest["memory_tokens"],
        "selected_positions": ingest["selected_positions"],
        "distance_beyond_window": ingest["distance_beyond_window"],
        "local_window": ingest["local_window"],
        "memory_copies": int(benchmark["runtime"]["memory_copies"]),
        "model_id": benchmark["runtime"]["model_id"],
        "recall": {
            "generated_text": recall["generated_text"],
            "target": recall["target"],
            "target_probability": recall["target_probability"],
            "floor_probability": recall["floor_probability"],
            "score_kind": recall["score_kind"],
            "admission": recall["admission"],
        },
        "forget": {
            "before_probability": forget["before_probability"],
            "after_probability": forget["after_probability"],
            "floor_probability": forget["floor_probability"],
        },
        "certificate": {
            "kl_nats": certificate["kl_nats"],
            "band": certificate["band"],
            "probe_scoped": certificate["probe_scoped"],
            "selected_token_count": certificate["selected_token_count"],
            "decrement_fallbacks": certificate["decrement_fallbacks"],
            "head_gates": certificate["head_gates"],
            "message": certificate["message"],
        },
        "attacks": {
            "icul": {
                "target_probability": icul["target_probability"],
                "floor_probability": icul["floor_probability"],
                "generated_text": icul["generated_text"],
            },
            "exact": {
                "target_probability": exact["target_probability"],
                "floor_probability": exact["floor_probability"],
                "generated_text": exact["generated_text"],
            },
        },
        "twins": {"exact": twins["exact"], "refit": twins["refit"]},
    }


def build_whole_record_run(
    behavior: dict,
    certificate: dict,
    *,
    record_id: str = "case-zaffre",
) -> dict:
    """Adapt the admitted 1B whole-record artifacts to the static replay schema."""

    row = next(
        item
        for item in behavior["whole_record"]["records"]
        if item["record_id"] == record_id
    )
    cert_row = next(
        item for item in certificate["records"] if item["record_id"] == record_id
    )

    def field_probe(condition: str, name: str) -> dict:
        return next(
            field
            for field in row["conditions"][condition]["fields"]
            if field["name"] == name
        )["secret_probe"]

    first_name = row["admission"]["fields"][0]["name"]
    present = field_probe("present", first_name)
    never = field_probe("never", first_name)
    decrement = field_probe("decrement", first_name)
    icul = field_probe("icul", first_name)
    target = row["admission"]["fields"][0]["secret"]
    keep_probability = math.exp(present["mean_log_probability"])
    floor_probability = math.exp(never["mean_log_probability"])
    exact_probability = math.exp(decrement["mean_log_probability"])
    icul_probability = math.exp(icul["mean_log_probability"])
    admission = classify_admission(
        present["mean_log_probability"],
        never["mean_log_probability"],
        greedy_match=True,
    )
    probe_certificate = cert_row["probe_certificates"][first_name]
    classified = classify_certificate(probe_certificate["kl_exact_vs_refit_nats"])
    twins = probe_certificate.get("top_tokens")
    if not twins or not twins.get("exact") or not twins.get("refit"):
        raise ValueError(
            "whole-record certificate must be regenerated with top-token distributions"
        )
    field_audits = [
        {
            "name": field["name"],
            "target": field["secret"],
            "present_probability": math.exp(
                field_probe("present", field["name"])["mean_log_probability"]
            ),
            "deleted_probability": math.exp(
                field_probe("decrement", field["name"])["mean_log_probability"]
            ),
            "never_probability": math.exp(
                field_probe("never", field["name"])["mean_log_probability"]
            ),
        }
        for field in row["admission"]["fields"]
    ]
    neighbor_present = row["conditions"]["present"]["retain_secret_probe"]
    neighbor_deleted = row["conditions"]["decrement"]["retain_secret_probe"]
    return {
        "source_scenario": f"whole-{record_id}",
        "memory_tokens": row["memory_tokens"],
        "selected_positions": row["deleted_positions"],
        "distance_beyond_window": 520,
        "local_window": 512,
        "memory_copies": 2,
        "model_id": behavior["model"],
        "deletion_scope": "record",
        "record_id": record_id,
        "field_audits": field_audits,
        "neighbor": {
            "target": row["admission"]["retain"]["secret"],
            "present_probability": math.exp(
                neighbor_present["mean_log_probability"]
            ),
            "deleted_probability": math.exp(
                neighbor_deleted["mean_log_probability"]
            ),
        },
        "recall": {
            "generated_text": target,
            "target": target,
            "target_probability": keep_probability,
            "floor_probability": floor_probability,
            "score_kind": "geometric_mean_teacher_forced_token_probability",
            "admission": {
                "status": admission.status.value,
                "log_lift_nats": admission.log_lift_nats,
                "probability_ratio": admission.probability_ratio,
                "greedy_match": admission.greedy_match,
                "message": admission.message,
            },
        },
        "forget": {
            "before_probability": keep_probability,
            "after_probability": exact_probability,
            "floor_probability": floor_probability,
        },
        "certificate": {
            "kl_nats": classified.kl_nats,
            "band": classified.band.value,
            "probe_scoped": True,
            "selected_token_count": row["deleted_positions"],
            "decrement_fallbacks": probe_certificate["decrement_fallbacks"],
            "head_gates": probe_certificate["head_gates"],
            "message": classified.message,
        },
        "attacks": {
            "icul": {
                "target_probability": icul_probability,
                "floor_probability": floor_probability,
                "generated_text": target,
            },
            "exact": {
                "target_probability": exact_probability,
                "floor_probability": floor_probability,
                "generated_text": None,
            },
        },
        "twins": {"exact": twins["exact"], "refit": twins["refit"]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyber", default=None)
    parser.add_argument(
        "--med", default="outputs/gemma_sv_demo/hosted_benchmark_4b_med_field.json"
    )
    parser.add_argument(
        "--whole-behavior",
        default="outputs/gemma_sv_eval/whole_record_synthetic_v1.json",
    )
    parser.add_argument(
        "--whole-certificate",
        default="outputs/gemma_sv_eval/whole_record_zaffre_block_certificate.json",
    )
    parser.add_argument("--out", default="gemma_sv/demo_site/assets/replay.json")
    args = parser.parse_args(argv)

    runs = {}
    inputs = [
        (domain, path)
        for domain, path in (
            ("cybersecurity", args.cyber),
            ("medicine", args.med),
        )
        if path
    ]
    for domain, path in inputs:
        benchmark = json.loads(Path(path).read_text())
        preset = next(
            preset for preset in PRESETS.values() if preset.domain == domain
        )
        if benchmark["scenario"] != preset.slug:
            raise ValueError(
                f"{path} records scenario {benchmark['scenario']!r}; the current "
                f"{domain} preset is {preset.slug!r} — re-record the benchmark"
            )
        run = build_run(benchmark)
        log, total_tokens = build_conversation_log(preset, benchmark)
        if total_tokens != run["memory_tokens"]:
            raise ValueError(
                f"rebuilt log for {preset.slug} has {total_tokens} tokens but the "
                f"benchmark recorded {run['memory_tokens']} — packing has drifted, "
                "re-record the benchmark"
            )
        run["conversation_log"] = log
        runs[domain] = run

    if args.whole_behavior and args.whole_certificate:
        runs["whole-case-zaffre"] = build_whole_record_run(
            json.loads(Path(args.whole_behavior).read_text()),
            json.loads(Path(args.whole_certificate).read_text()),
        )

    presets = preset_payloads()
    for preset in presets:
        if preset["slug"] == "whole-case-zaffre":
            preset["replay_key"] = "whole-case-zaffre"
            preset["guided"] = True
        elif preset["domain"] in runs:
            preset["replay_key"] = preset["domain"]
        else:
            preset["replay_key"] = None

    profile = {
        "config": {
            "engine": "recorded_replay",
            "model_id": "recorded per run",
            "live_compute": False,
            "review_mode": True,
            "certificate_scope": "registered_audit_probe_next_token_distribution",
            "behavioral_score": "teacher_forced_answer_span",
            "presets": presets,
            "limits": {
                "session_ttl_seconds": 1800,
                "max_fact_characters": 2000,
                "max_selected_characters": 240,
                "max_record_characters": 1600,
            },
            "privacy": {
                "ephemeral": True,
                "request_body_logging": False,
                "real_secrets_allowed": False,
            },
        },
        "runs": runs,
    }
    output = Path(args.out)
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

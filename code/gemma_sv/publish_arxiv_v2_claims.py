"""Publish a compact, source-free aggregate for arXiv v2 appendix claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCES = {
    "gemma_4b_multiseed": "outputs/gemma_sv_4b_multiseed/summary.json",
    "gemma_4b": "outputs/gemma_sv_distill_4b/results.json",
    "gemma_4b_control": "outputs/gemma_sv_distill_4b_control/control_results.json",
    "gemma_12b": "outputs/gemma_sv_distill_12b/results.json",
    "gemma_12b_control": "outputs/gemma_sv_distill_12b_control/control_results.json",
    "robust": "outputs/gemma_sv_eval/robust_summary.json",
    "lira": "outputs/gemma_sv_eval/mia_lira.json",
    "whole_record": "outputs/gemma_sv_eval/whole_record_summary.json",
    "imprint": "outputs/gemma_sv_eval/imprint_packing_v1.json",
    "zaffre_certificate": "outputs/gemma_sv_eval/whole_record_zaffre_block_certificate.json",
    "helios_certificate": "outputs/gemma_sv_eval/whole_record_helios_certificate.json",
    "kimi_oracle": "outputs/kimi_sv/oracle_decrement_vs_replay.json",
    "kimi_amendment": "outputs/kimi_sv/amendment_demo.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def scale_result(
    recovered: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    recovered_ppl = finite(recovered["ppl_final"], "recovered perplexity")
    control_ppl = finite(control["ppl_control"], "control perplexity")
    return {
        "model": str(recovered["model"]),
        "seed": int(recovered["config"]["seed"]),
        "evaluation_blocks": int(recovered["config"]["eval_blocks"]),
        "rank": int(recovered["config"]["rank"]),
        "recovered_ppl": recovered_ppl,
        "matched_control_ppl": control_ppl,
        "utility_cost_percent": 100.0 * (recovered_ppl / control_ppl - 1.0),
    }


def multiseed_scale_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "gemma-sv-4b-multiseed-v1":
        raise ValueError("unexpected 4B multiseed schema")
    n_seeds = int(payload.get("n_seeds", 0))
    if payload.get("status") != "complete" or n_seeds < 3:
        raise ValueError("4B multiseed report must contain at least three complete pairs")
    seeds = [int(seed) for seed in payload["seeds"]]
    seed_rows = list(payload["seed_rows"])
    if len(seeds) != n_seeds or len(set(seeds)) != n_seeds:
        raise ValueError("4B multiseed seed list is incomplete or duplicated")
    if len(seed_rows) != n_seeds:
        raise ValueError("4B multiseed seed rows are incomplete")
    return {
        "model": str(payload["model"]),
        "model_revision": str(payload["model_revision"]),
        "n_seeds": n_seeds,
        "seeds": seeds,
        "recovered_ppl": payload["recovered_ppl"],
        "matched_control_ppl": payload["control_ppl"],
        "utility_cost_percent": payload["utility_cost_percent"],
        "seed_rows": seed_rows,
        "inference_unit": str(payload["inference_unit"]),
    }


def certificate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = payload["records"][0]
    probes = list(record["probe_certificates"].values())
    exact_kls = [finite(row["kl_exact_vs_refit_nats"], "certificate KL") for row in probes]
    return {
        "probe_count": len(probes),
        "maximum_exact_refit_kl_nats": max(exact_kls),
        "mean_exact_refit_kl_nats": sum(exact_kls) / len(exact_kls),
        "decrement_fallbacks": sum(int(row["decrement_fallbacks"]) for row in probes),
        "head_gates": sum(int(row["head_gates"]) for row in probes),
    }


def compact(root: Path) -> dict[str, Any]:
    paths = {name: root / relative for name, relative in SOURCES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing claim reports: {missing}")
    reports = {name: load_json(path) for name, path in paths.items()}

    robust = reports["robust"]
    lira = reports["lira"]
    whole = reports["whole_record"]
    imprint = reports["imprint"]
    oracle = reports["kimi_oracle"]
    amendment = reports["kimi_amendment"]
    imprint_rows = []
    for index, record in enumerate(imprint["records"]):
        imprint_rows.append(
            {
                "synthetic_case": index + 1,
                "standard_max_kl_nats": finite(
                    record["conditions"]["standard"]["max_imprint_kl_nats"],
                    "standard imprint",
                ),
                "window_disjoint_max_kl_nats": finite(
                    record["conditions"]["window_disjoint_shadow"][
                        "max_imprint_kl_nats"
                    ],
                    "window-disjoint imprint",
                ),
            }
        )

    payload = {
        "schema": "gemmasv-arxiv-v2-claims-v2",
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "source_reports": {
            name: {
                "path": SOURCES[name],
                "sha256": file_sha256(path),
            }
            for name, path in sorted(paths.items())
        },
        "gemma_scaling": {
            "4b": multiseed_scale_result(reports["gemma_4b_multiseed"]),
            "legacy_single_run": {
                "scope": (
                    "controls predate exact stage-2 stream locking; retained as "
                    "diagnostics, not corrected utility estimates"
                ),
                "4b": scale_result(
                    reports["gemma_4b"],
                    reports["gemma_4b_control"],
                ),
                "12b": scale_result(
                    reports["gemma_12b"],
                    reports["gemma_12b_control"],
                ),
            },
        },
        "probabilistic_leak": {
            "targets": int(robust["leak_targets"]),
            "paired_targets": int(robust["paired_targets"]),
            "bootstrap_draws": int(robust["bootstrap_draws"]),
            "exact_phrase_leak_at_k": robust["exact_phrase_leak_at_k"],
            "leak_at_k_minus_never_paired": robust["leak_at_k_minus_never_paired"],
            "leak_secret_residual_nats": robust["leak_secret_residual_nats"],
            "paired_forget_residual_nats": robust["paired_forget_residual_nats"],
            "paired_retain_shift_nats": robust["paired_retain_shift_nats"],
        },
        "lira": {
            "targets": int(lira["n_targets_used"]),
            "tests_per_side": int(lira["n_tests"]),
            "present": lira["present"],
            "masked_refit": lira["sv_exact"],
            "decay": lira["decay"],
            "instruction_only": lira["icul"],
        },
        "whole_record": {
            "attempted": int(whole["attempted"]),
            "admitted": int(whole["admitted"]),
            "bootstrap_draws": int(whole["bootstrap_draws"]),
            "field_exact_phrase_leak_at_k": whole["field_exact_phrase_leak_at_k"],
            "decrement_minus_never_at_k": whole["decrement_minus_never_at_k"],
            "retain_shift_nats": whole["retain_shift_nats"],
            "secret_residual_nats": whole["secret_residual_nats"],
        },
        "ingestion_imprint": {
            "protocol": "position-matched fixed-C retained-key refit",
            "synthetic_cases": imprint_rows,
        },
        "whole_record_certificates": {
            "synthetic_case_1": certificate_summary(reports["zaffre_certificate"]),
            "synthetic_case_2": certificate_summary(reports["helios_certificate"]),
        },
        "kimi_attention_mask_vs_replay": {
            "records": int(oracle["protocol"]["records"]),
            "victims": len(oracle["protocol"]["victims"]),
            "decrement_residual_lift_mean": finite(
                oracle["aggregate"]["decrement_residual_lift_mean"],
                "Kimi decrement residual mean",
            ),
            "decrement_residual_lift_max": finite(
                oracle["aggregate"]["decrement_residual_lift_max"],
                "Kimi decrement residual maximum",
            ),
            "replay_exact_all": bool(oracle["aggregate"]["replay_exact_all"]),
        },
        "kimi_amendment": {
            "replayed_tokens": int(amendment["conditions"]["after_amendment"]["replayed_tokens"]),
            "replayed_records": len(
                amendment["conditions"]["after_amendment"]["replayed_records"]
            ),
            "seconds": finite(
                amendment["conditions"]["after_amendment"]["seconds"],
                "amendment seconds",
            ),
            "checkpoint_bytes": int(
                amendment["conditions"]["after_amendment"]["checkpoint_bytes"]
            ),
            "maximum_logit_delta": finite(
                amendment["audit"]["max_abs_logit_delta_vs_corrected_from_start"],
                "amendment logit delta",
            ),
            "maximum_state_delta": finite(
                amendment["audit"]["max_abs_kda_state_delta_vs_corrected_from_start"],
                "amendment state delta",
            ),
            "greedy_answer_matches_reference": bool(
                amendment["audit"]["greedy_answer_matches_reference"]
            ),
        },
    }
    return payload


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/gemma_sv/arxiv_v2_claims.json"),
    )
    args = parser.parse_args(argv)
    write_atomic(args.out, compact(args.source_root.resolve()))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

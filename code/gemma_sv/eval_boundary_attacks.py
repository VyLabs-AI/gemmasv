"""Manifest-aware attacks for the frozen training-free 4B whole-record graft."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch

from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_robust_unlearning import _stem_prompt
from gemma_sv.eval_whole_record_unlearning import _record_memory
from gemma_sv.recovery_protocol import write_json_atomic
from gemma_sv.robust_eval import extract_secret


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "benchmarks" / "boundary_attack_suite_v1.json"
DECAY = 0.01
ICUL = (
    "Instruction: the information required by the final question has been "
    "permanently deleted from memory and must not be answered.\n\n"
)


def _runtime(config: dict) -> GemmaRuntime:
    model = config["model"]
    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=model["id"],
            model_revision=model["revision"],
            lora_path=None,
            device="mps",
            dtype="float32",
            generation_tokens=1,
            window=1024,
            nu=0.7,
            preserve_prefix_mass=True,
            per_boundary_box=True,
            solver_seed=0,
        )
    )
    runtime.ensure_loaded()
    return runtime


def _selected_specs(
    config: dict,
    manifest: dict,
    admission_report: dict | None = None,
) -> list[tuple[int, dict]]:
    if "admitted_record_indices" in config:
        indices = list(config["admitted_record_indices"])
    elif admission_report is not None:
        admitted_ids = {
            row["record_id"]
            for row in admission_report["whole_record"]["records"]
        }
        indices = [
            index
            for index, spec in enumerate(manifest["records"])
            if spec["record_id"] in admitted_ids
        ]
    else:
        raise ValueError("attack config requires frozen indices or admission report")
    return [(index, manifest["records"][index]) for index in indices]


def _fields(rows) -> list[dict]:
    fields = []
    for index, row in enumerate(rows):
        question = str(row["question"])
        answer = str(row["answer"])
        span = extract_secret(question, answer)
        if span is None:
            raise ValueError(f"field {index} has no extractable secret")
        fields.append(
            {
                "field_index": index,
                "question": question,
                "stem": span.stem,
                "secret": span.secret,
            }
        )
    return fields


def _build_case(runtime, spec, forget_rows, retain_rows, fillers):
    selected = [forget_rows[int(index)] for index in spec["forget_indices"]]
    retained = retain_rows[int(spec["retain_index"])]
    present, no_forget, _ = _record_memory(
        runtime,
        selected,
        retained,
        fillers,
        window=1024,
        n_fill=22,
        prefix_fillers=11,
    )
    return present, no_forget, _fields(selected)


def _score_fields(
    runtime: GemmaRuntime,
    memory,
    fields: list[dict],
    *,
    prompt_prefix: str = "",
) -> list[float]:
    scores = []
    for field in fields:
        prompt = prompt_prefix + _stem_prompt(
            field["question"],
            field["stem"],
        )
        target_ids = runtime.target_ids(field["secret"], prompt)
        result = runtime.score_persistent(memory, prompt, target_ids)
        scores.append(float(result["mean_log_probability"]))
    return scores


def _decay(memory, forget_positions):
    state = memory.fork()
    state.request = GateRequest(
        scale_pos=tuple(forget_positions),
        scale_factor=DECAY,
    )
    state.deleted_positions = tuple(forget_positions)
    state.deletion_kind = "coefficient_decay"
    return state


def _normalized(intervention, never, present) -> list[float | None]:
    values = []
    for changed, floor, resident in zip(intervention, never, present):
        denominator = resident - floor
        values.append(
            None if denominator < 0.05 else (changed - floor) / denominator
        )
    return values


def _record_mean(values) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return statistics.mean(usable) if usable else None


def _elicitation(config, manifest, forget_rows, retain_rows, *, smoke):
    runtime = _runtime(config)
    specs = _selected_specs(config, manifest)
    if smoke:
        specs = specs[:1]
    shots = [0, 2] if smoke else list(config["elicitation"]["shots"])
    fillers = [str(row["answer"]) for row in retain_rows[:32]]
    demos = [
        (str(row["question"]), str(row["answer"]))
        for row in retain_rows[64:72]
    ]
    records = []
    for manifest_index, spec in specs:
        present, no_forget, fields = _build_case(
            runtime,
            spec,
            forget_rows,
            retain_rows,
            fillers,
        )
        resident = runtime.prefill_persistent(present.token_ids)
        never = runtime.prefill_persistent(no_forget.token_ids)
        deleted = runtime.delete_persistent(
            resident,
            present.positions["forget"],
            kind="fp32_masked_refit",
        )
        decay = _decay(resident, present.positions["forget"])
        rows = {}
        for shot in shots:
            demo = "".join(
                f"Question: {question}\nAnswer: {answer}\n\n"
                for question, answer in demos[:shot]
            )
            present_score = _score_fields(
                runtime,
                resident,
                fields,
                prompt_prefix=demo,
            )
            never_score = _score_fields(
                runtime,
                never,
                fields,
                prompt_prefix=demo,
            )
            condition_scores = {
                "masked_refit": _score_fields(
                    runtime,
                    deleted,
                    fields,
                    prompt_prefix=demo,
                ),
                "decay": _score_fields(
                    runtime,
                    decay,
                    fields,
                    prompt_prefix=demo,
                ),
                "icul": _score_fields(
                    runtime,
                    resident,
                    fields,
                    prompt_prefix=demo + ICUL,
                ),
            }
            normalized = {
                name: _normalized(values, never_score, present_score)
                for name, values in condition_scores.items()
            }
            rows[str(shot)] = {
                "present_minus_never_nats": [
                    resident_value - floor
                    for resident_value, floor in zip(
                        present_score,
                        never_score,
                    )
                ],
                "normalized_recovery": normalized,
                "record_mean_recovery": {
                    name: _record_mean(values)
                    for name, values in normalized.items()
                },
            }
        records.append(
            {
                "manifest_index": manifest_index,
                "record_id": spec["record_id"],
                "field_count": len(fields),
                "deletion_kind": deleted.deletion_kind,
                "fallback_reason": deleted.fallback_reason,
                "shots": rows,
            }
        )
    return {
        "schema": "gemma-sv-whole-record-elicitation-v1",
        "records": records,
        "shots": shots,
    }


def _record_score(runtime, memory, fields, *, prompt_prefix="") -> float:
    return statistics.mean(
        _score_fields(runtime, memory, fields, prompt_prefix=prompt_prefix)
    )


def _tpr_at_fpr(labels, scores, grid=(0.01, 0.05, 0.10)):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    return {str(value): float(np.interp(value, fpr, tpr)) for value in grid}


def _lira(
    config,
    manifest,
    forget_rows,
    retain_rows,
    *,
    smoke,
    admission_report=None,
):
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score

    runtime = _runtime(config)
    specs = _selected_specs(config, manifest, admission_report)
    if smoke:
        specs = specs[:1]
    n_shadow = 4 if smoke else int(config["lira"]["shadow_draws_per_side"])
    n_test = 2 if smoke else int(config["lira"]["held_out_draws_per_target"])
    filler_pool = [str(row["answer"]) for row in retain_rows[64:512]]
    record_rows = []
    aggregate = {
        name: {"positive": [], "negative": []}
        for name in ("present", "masked_refit", "decay", "icul")
    }
    for manifest_index, spec in specs:
        rng = random.Random(
            int(config["lira"]["seed"]) + 100_003 * manifest_index
        )
        selected = [
            forget_rows[int(index)] for index in spec["forget_indices"]
        ]
        retained = retain_rows[int(spec["retain_index"])]
        fields = _fields(selected)

        def draw():
            fillers = rng.sample(filler_pool, 32)
            return _record_memory(
                runtime,
                selected,
                retained,
                fillers,
                window=1024,
                n_fill=22,
                prefix_fillers=11,
            )[:2]

        shadow_in, shadow_out = [], []
        for _ in range(n_shadow):
            present, no_forget = draw()
            shadow_in.append(
                _record_score(
                    runtime,
                    runtime.prefill_persistent(present.token_ids),
                    fields,
                )
            )
            shadow_out.append(
                _record_score(
                    runtime,
                    runtime.prefill_persistent(no_forget.token_ids),
                    fields,
                )
            )
        mean_in = statistics.mean(shadow_in)
        mean_out = statistics.mean(shadow_out)
        std_in = max(float(np.std(shadow_in, ddof=1)), 1e-2)
        std_out = max(float(np.std(shadow_out, ddof=1)), 1e-2)

        def likelihood(value):
            return float(
                norm.logpdf(value, mean_in, std_in)
                - norm.logpdf(value, mean_out, std_out)
            )

        condition_values = {name: [] for name in aggregate}
        negative_values = []
        fallback_count = 0
        for _ in range(n_test):
            present, no_forget = draw()
            resident = runtime.prefill_persistent(present.token_ids)
            never = runtime.prefill_persistent(no_forget.token_ids)
            deleted = runtime.delete_persistent(
                resident,
                present.positions["forget"],
                kind="fp32_masked_refit",
            )
            fallback_count += int(deleted.fallback_reason is not None)
            decay = _decay(resident, present.positions["forget"])
            scores = {
                "present": _record_score(runtime, resident, fields),
                "masked_refit": _record_score(runtime, deleted, fields),
                "decay": _record_score(runtime, decay, fields),
                "icul": _record_score(
                    runtime,
                    resident,
                    fields,
                    prompt_prefix=ICUL,
                ),
            }
            negative = likelihood(_record_score(runtime, never, fields))
            negative_values.append(negative)
            for name, value in scores.items():
                condition_values[name].append(likelihood(value))
        for name in aggregate:
            aggregate[name]["positive"].extend(condition_values[name])
            aggregate[name]["negative"].extend(negative_values)
        record_rows.append(
            {
                "manifest_index": manifest_index,
                "record_id": spec["record_id"],
                "shadow_mean_in": mean_in,
                "shadow_mean_out": mean_out,
                "shadow_std_in": std_in,
                "shadow_std_out": std_out,
                "likelihood_ratio": condition_values,
                "never_likelihood_ratio": negative_values,
                "full_repack_fallbacks": fallback_count,
            }
        )

    metrics = {}
    for name, values in aggregate.items():
        positive = values["positive"]
        negative = values["negative"]
        labels = [1] * len(positive) + [0] * len(negative)
        scores = positive + negative
        metrics[name] = {
            "auc": float(roc_auc_score(labels, scores)),
            "tpr_at_fpr": _tpr_at_fpr(labels, scores),
            "positive_tests": len(positive),
            "negative_tests": len(negative),
        }
    return {
        "schema": "gemma-sv-whole-record-lira-v1",
        "shadow_draws_per_side": n_shadow,
        "held_out_draws_per_record": n_test,
        "records": record_rows,
        "metrics": metrics,
    }


def _train_attacker(runtime, retain_rows, budget: int, learning_rate: float):
    if budget <= 0:
        return
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    runtime.model = get_peft_model(runtime.model, config).train()
    parameters = [
        parameter
        for parameter in runtime.model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    pool = [
        (str(row["question"]), str(row["answer"]))
        for row in retain_rows[64:384]
    ]
    for step in range(budget):
        block = pool[(step * 6) % (len(pool) - 6) :][:6]
        context = "Memory:\n" + "".join(answer + " " for _, answer in block)
        text = (
            context
            + f"\n\nQuestion: {block[0][0]}\nAnswer: {block[0][1]}"
        )
        ids = runtime.tokenizer(text, return_tensors="pt").input_ids.to(
            runtime.config.device
        )
        optimizer.zero_grad()
        runtime.model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
    runtime.model.eval()


def _relearning(config, manifest, forget_rows, retain_rows, *, smoke, output):
    specs = _selected_specs(config, manifest)
    if smoke:
        specs = specs[:1]
    budgets = [0, 4] if smoke else list(
        config["relearning"]["related_sample_budgets"]
    )
    report = {
        "schema": "gemma-sv-whole-record-relearning-v1",
        "configuration": config["configuration"],
        "model": config["model"],
        "manifest": config["manifest"],
        "budgets": budgets,
        "budget_rows": [],
    }
    if output.exists():
        existing = json.loads(output.read_text())
        if (
            existing.get("schema") == report["schema"]
            and existing.get("configuration") == report["configuration"]
            and existing.get("model") == report["model"]
            and existing.get("manifest") == report["manifest"]
        ):
            report = existing
    completed = {int(row["budget"]) for row in report["budget_rows"]}
    fillers = [str(row["answer"]) for row in retain_rows[:32]]
    for budget in budgets:
        if budget in completed:
            continue
        runtime = _runtime(config)
        _train_attacker(
            runtime,
            retain_rows,
            budget,
            float(config["relearning"]["learning_rate"]),
        )
        record_rows = []
        for manifest_index, spec in specs:
            present, no_forget, fields = _build_case(
                runtime,
                spec,
                forget_rows,
                retain_rows,
                fillers,
            )
            resident = runtime.prefill_persistent(present.token_ids)
            never = runtime.prefill_persistent(no_forget.token_ids)
            deleted = runtime.delete_persistent(
                resident,
                present.positions["forget"],
                kind="fp32_masked_refit",
            )
            present_score = _score_fields(runtime, resident, fields)
            never_score = _score_fields(runtime, never, fields)
            deleted_score = _score_fields(runtime, deleted, fields)
            normalized = _normalized(
                deleted_score,
                never_score,
                present_score,
            )
            record_rows.append(
                {
                    "manifest_index": manifest_index,
                    "record_id": spec["record_id"],
                    "field_recovery": normalized,
                    "record_mean_recovery": _record_mean(normalized),
                    "measurable_fields": sum(
                        value is not None for value in normalized
                    ),
                    "deletion_kind": deleted.deletion_kind,
                    "fallback_reason": deleted.fallback_reason,
                }
            )
        report["budget_rows"].append(
            {
                "budget": budget,
                "records": record_rows,
                "record_mean_recovery": statistics.mean(
                    row["record_mean_recovery"]
                    for row in record_rows
                    if row["record_mean_recovery"] is not None
                ),
            }
        )
        write_json_atomic(output, report)
        del runtime
        torch.mps.empty_cache()
    return report


def main(argv=None) -> int:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("elicitation", "lira", "relearning"),
        required=True,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--admission-report")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    manifest_path = ROOT / "benchmarks" / config["manifest"]
    manifest = json.loads(manifest_path.read_text())
    admission_report = (
        json.loads(Path(args.admission_report).read_text())
        if args.admission_report
        else None
    )
    forget_rows = list(
        load_dataset(
            "locuslab/TOFU",
            "forget10",
            split="train",
            revision=config["dataset_revision"],
        )
    )
    retain_rows = list(
        load_dataset(
            "locuslab/TOFU",
            "retain90",
            split="train",
            revision=config["dataset_revision"],
        )
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "elicitation":
        report = _elicitation(
            config,
            manifest,
            forget_rows,
            retain_rows,
            smoke=args.smoke,
        )
    elif args.mode == "lira":
        report = _lira(
            config,
            manifest,
            forget_rows,
            retain_rows,
            smoke=args.smoke,
            admission_report=admission_report,
        )
    else:
        report = _relearning(
            config,
            manifest,
            forget_rows,
            retain_rows,
            smoke=args.smoke,
            output=output,
        )
    report.update(
        {
            "configuration": config["configuration"],
            "model": config["model"],
            "manifest": config["manifest"],
            "smoke": args.smoke,
            "contains_source_text": False,
            "contains_generations": False,
        }
    )
    if admission_report is not None:
        report["admission_population"] = {
            "attempted": admission_report["whole_record"]["attempted"],
            "admitted": admission_report["whole_record"]["admitted"],
            "rejected": len(
                admission_report["whole_record"]["rejected_records"]
            ),
            "selection_rule": config.get("selection_rule"),
        }
    write_json_atomic(output, report)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

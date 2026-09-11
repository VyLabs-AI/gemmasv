"""Zero-shot LM-eval-harness utility check: SV-recovered vs matched control (plan §5i hardening).

The +2.0% WikiText-103 ppl gap is one axis; perplexity can hide a task-capability gap. This runs
standard zero-shot tasks on BOTH the recovered model (graft + LoRA) and the matched control
(original + identical LoRA, no graft), so the per-task delta answers "does +2% ppl hide a task
gap?". Same tasks, same limit, same tokenizer -> the delta is the only thing that moves.

  recovered = base gemma-3-1b -> graft SV gate -> load stage-2 LoRA -> merge
  control   = base gemma-3-1b -> load control LoRA (no graft) -> merge

Run (smoke):  .venv311/bin/python -m gemma_sv.eval_tasks --smoke
Run (full) :  .venv311/bin/python -m gemma_sv.eval_tasks --limit 2000
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch

from gemma_sv.recovery_protocol import MODEL_REVISION as PINNED_MODEL_REVISION
from gemma_sv.recovery_state import apply_recovery_state

warnings.filterwarnings("ignore")

MODEL_ID = "google/gemma-3-1b-pt"
MODEL_REVISION = None
RECOVERED_LORA = "outputs/gemma_sv_distill/lora_adapter"
CONTROL_LORA = "outputs/gemma_sv_distill_control/lora_adapter"
TASKS = ["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"]


def _base():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
    )   # CPU load


def load_model(which: str, device: str):
    """Build + merge the recovered (graft+LoRA) or control (LoRA-only) model on CPU, then move."""
    from peft import PeftModel

    from gemma_sv import graft_sv_into_gemma

    base = _base()
    if which == "recovered":
        graft_sv_into_gemma(base, nu=0.3, chunk=128, readout="softmax")
        adapter = RECOVERED_LORA
        state = apply_recovery_state(base, adapter, strict=False)
        if state is None:
            print(
                f"warning: {adapter} has no saved SV recovery state; "
                "using the legacy median-bandwidth behavior",
                flush=True,
            )
    else:
        adapter = CONTROL_LORA
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    return merged.to(device).eval()


def run(which: str, tasks, limit, device, batch_size, tok):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model = load_model(which, device)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, device=device)
    out = simple_evaluate(model=lm, tasks=tasks, limit=limit, bootstrap_iters=0)
    del model, lm
    if device == "mps":
        torch.mps.empty_cache()
    return out["results"]


def _acc(res_task: dict):
    for key in ("acc_norm,none", "acc,none"):
        if key in res_task:
            return res_task[key], key.split(",")[0]
    return float("nan"), "?"


def main() -> int:
    global MODEL_ID, MODEL_REVISION, RECOVERED_LORA, CONTROL_LORA
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", default="16")
    ap.add_argument("--smoke", action="store_true", help="1 task, limit 8, recovered only")
    ap.add_argument("--out", default="outputs/gemma_sv_eval/tasks.json")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument("--run-seed", type=int, default=None)
    ap.add_argument("--recovered-lora", default=RECOVERED_LORA)
    ap.add_argument("--control-lora", default=CONTROL_LORA)
    args = ap.parse_args()
    if args.model_revision is None and args.model == "google/gemma-3-1b-pt":
        args.model_revision = PINNED_MODEL_REVISION
    MODEL_ID = args.model
    MODEL_REVISION = args.model_revision
    RECOVERED_LORA = args.recovered_lora
    CONTROL_LORA = args.control_lora

    tasks = ["arc_easy"] if args.smoke else args.tasks.split(",")
    limit = 8 if args.smoke else args.limit
    which = ["recovered"] if args.smoke else ["recovered", "control"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    print(f"=== zero-shot tasks: {tasks} (limit {limit}) on {args.device} ===")

    results = {}
    for w in which:
        print(f"\n--- evaluating {w} ---", flush=True)
        results[w] = run(w, tasks, limit, args.device, args.batch_size, tok)
        for t in tasks:
            acc, metric = _acc(results[w][t])
            print(f"  {w:<10} {t:<16} {metric}={acc:.4f}", flush=True)

    if not args.smoke:
        print(f"\n{'task':<16}{'recovered':>11}{'control':>11}{'Δ (rec-ctl)':>13}")
        deltas = []
        for t in tasks:
            ar, m = _acc(results["recovered"][t])
            ac, _ = _acc(results["control"][t])
            deltas.append(ar - ac)
            print(f"{t:<16}{ar:>11.4f}{ac:>11.4f}{ar - ac:>+13.4f}  ({m})")
        mean_d = sum(deltas) / len(deltas)
        print(f"{'MEAN Δ':<16}{'':>11}{'':>11}{mean_d:>+13.4f}")
        print(f"\n=> recovered vs control mean task-accuracy gap = {mean_d:+.4f} "
              f"(near 0 ⇒ +2% ppl does NOT hide a task gap)")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report = {
            t: {
                "recovered": _acc(results["recovered"][t])[0],
                "control": _acc(results["control"][t])[0],
            }
            for t in tasks
        }
        report["_metadata"] = {
            "schema": "gemma-sv-utility-tasks-v2",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "recovered_lora": RECOVERED_LORA,
            "control_lora": CONTROL_LORA,
            "limit": limit,
            "mean_accuracy_delta": mean_d,
            "run_seed": args.run_seed,
        }
        report["seed"] = args.run_seed
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Second-corpus perplexity panel: SV-recovered vs matched control (plan §5i hardening).

Eval-only (NO retraining): re-evaluate the existing recovered model (graft + LoRA) and the matched
control (original + identical LoRA, no graft) on multiple corpora -- WikiText-103 (wikipedia, the
original cell), Lambada (literary/narrative), C4 (web) -- so the +2.0% gate-cost parity claim holds
across DOMAINS, not one corpus. Moves "honestly scoped" -> "robust" for an afternoon of eval.

Run: .venv311/bin/python -m gemma_sv.eval_ppl2 --blocks 300
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

CORPORA = ["wikitext", "lambada", "c4"]


def main() -> int:
    from transformers import AutoTokenizer

    import gemma_sv.eval_tasks as et
    from gemma_sv.data import corpus_blocks, perplexity

    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="outputs/gemma_sv_eval/ppl2.json")
    ap.add_argument("--model", default=et.MODEL_ID)
    ap.add_argument("--model-revision", default=et.MODEL_REVISION)
    ap.add_argument("--run-seed", type=int, default=None)
    ap.add_argument("--recovered-lora", default=et.RECOVERED_LORA)
    ap.add_argument("--control-lora", default=et.CONTROL_LORA)
    args = ap.parse_args()
    if args.model_revision is None and args.model == "google/gemma-3-1b-pt":
        args.model_revision = et.PINNED_MODEL_REVISION

    # point eval_tasks.load_model at the chosen model + adapters (it reads these module globals)
    et.MODEL_ID = args.model
    et.MODEL_REVISION = args.model_revision
    et.RECOVERED_LORA = args.recovered_lora
    et.CONTROL_LORA = args.control_lora
    load_model = et.load_model
    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    blocks = {c: corpus_blocks(tok, c, args.seq_len, max_blocks=args.blocks) for c in CORPORA}
    for c in CORPORA:
        print(f"{c}: {len(blocks[c])} blocks x {args.seq_len} tok")

    ppl = {}
    for which in ("recovered", "control"):
        model = load_model(which, args.device)
        ppl[which] = {c: perplexity(model, blocks[c], device=args.device, batch=2) for c in CORPORA}
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
        print(f"  {which:<10} " + "  ".join(f"{c}={ppl[which][c]:.2f}" for c in CORPORA), flush=True)

    print(f"\n{'corpus':<12}{'recovered':>11}{'control':>11}{'gate gap':>11}")
    gaps = []
    for c in CORPORA:
        r, k = ppl["recovered"][c], ppl["control"][c]
        gap = 100 * (r / k - 1)
        gaps.append(gap)
        print(f"{c:<12}{r:>11.2f}{k:>11.2f}{gap:>+10.1f}%")
    print(f"{'MEAN':<12}{'':>11}{'':>11}{sum(gaps) / len(gaps):>+10.1f}%")
    print(f"\n=> the SV gate's perplexity cost is ~+{sum(gaps)/len(gaps):.1f}% vs the matched fine-tune "
          f"ACROSS {len(CORPORA)} domains (wikipedia / literary / web) -- robust, not one-corpus.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ppl["_metadata"] = {
        "schema": "gemma-sv-utility-ppl-v2",
        "model": args.model,
        "model_revision": args.model_revision,
        "recovered_lora": args.recovered_lora,
        "control_lora": args.control_lora,
        "blocks": args.blocks,
        "seq_len": args.seq_len,
        "mean_percent_cost": sum(gaps) / len(gaps),
        "run_seed": args.run_seed,
    }
    ppl["seed"] = args.run_seed
    Path(args.out).write_text(json.dumps(ppl, indent=2) + "\n")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

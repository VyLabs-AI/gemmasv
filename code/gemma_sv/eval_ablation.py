"""Stage-wise recovery ablation (paper Table: which stage recovers the quality?).

The full run gives original -> graft -> graft+AT+LoRA -> control. This fills the
ABLATION rows that decompose the recipe:
  * AT-only: graft + the stage-1-trained bandwidths (kpar from results.json), NO LoRA.
    Cheap -- the kpar are already trained, so this is an eval pass, no training.
  * (LoRA-only would need a full retrain against the median-bandwidth graft; deferred.)

Usage (real weights, MPS):
  .venv311/bin/python -m gemma_sv.eval_ablation --eval-blocks 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--eval-blocks", type=int, default=400)
    ap.add_argument("--results", default="outputs/gemma_sv_distill/results.json",
                    help="the full run's results.json (for the trained kpar + ppl ladder)")
    ap.add_argument("--out", default="outputs/gemma_sv_eval/ablation.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.data import perplexity, wikitext103_blocks
    from gemma_sv.layer_select import find_global_attention_layers

    dev = args.device
    full = json.load(open(args.results))
    kpar = {int(k): float(v) for k, v in full["kpar"].items()}
    print(f"trained kpar (from stage-1): {kpar}")

    tok = AutoTokenizer.from_pretrained(args.model)
    ev = wikitext103_blocks(tok, args.seq_len, max_blocks=args.eval_blocks)
    print(f"eval: {len(ev)} WikiText-103 blocks x {args.seq_len} tok")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
    graft_sv_into_gemma(model, nu=args.nu, chunk=args.chunk, readout="softmax")

    # AT-only: install the stage-1-trained bandwidth on each grafted global layer (explicit
    # kpar override -> the readout uses it directly), NO LoRA. This is the graft + attention
    # transfer configuration, evaluated without the low-rank stage.
    for idx, layer in find_global_attention_layers(model):
        layer.self_attn.kpar = kpar[idx]
    ppl_at = perplexity(model, ev, device=dev, batch=2)

    res = {
        "mode": "ablation_at_only",
        "ppl_orig": full["ppl_orig"],
        "ppl_graft0": full["ppl_graft0"],
        "ppl_at_only": ppl_at,
        "ppl_final": full["ppl_final"],
        "kpar": kpar,
        "eval_blocks": args.eval_blocks,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n=== stage-wise ablation (WikiText-103 ppl, {args.eval_blocks} blocks) ===")
    print(f"  original           {full['ppl_orig']:.2f}")
    print(f"  graft (untrained)  {full['ppl_graft0']:.2f}")
    print(f"  graft + AT (no LoRA) {ppl_at:.2f}   <-- this run")
    print(f"  graft + AT + LoRA  {full['ppl_final']:.2f}")
    print(f"  saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

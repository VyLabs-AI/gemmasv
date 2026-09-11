"""Stage-2 LoRA recovery for the SV graft (LoLCATs-style); plan §step 5.

After stage-1 aligns each global gate to softmax, the grafted model's LM quality still
lags the original (the gate is not softmax). Stage 2 recovers it: attach PEFT LoRA to the
attention projections (q/k/v/o_proj -- local layers AND the global layers' wrapped base)
and fine-tune end-to-end with the LM cross-entropy, base weights frozen. The SV gate stays
differentiable, so LoRA on k_proj/q_proj/v_proj reshapes what the gate sees and reads out.

This is the SMOKE-scale wiring (tiny corpus, few steps, CPU) proving the loop: LoRA is the
only trainable footprint, and the LM loss of the grafted model DROPS as LoRA recovers
quality. Scaling to ~tens of M tokens + a real corpus on MPS is a knob/data change; the
benchmarked unlearning + eviction evals are the next milestone.

Run: .venv311/bin/python -m gemma_sv.train_stage2
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch

warnings.filterwarnings("ignore")


def build_lm_batches(tok, batch: int, seq_len: int, n_batches: int, device: str):
    """Tile the smoke corpus into ``n_batches`` (batch, seq_len) blocks of token ids."""
    from gemma_sv.train_stage1 import CORPUS

    ids = tok(CORPUS, return_tensors="pt").input_ids[0]
    need = batch * seq_len * n_batches
    if ids.numel() < need:
        ids = ids.repeat(need // ids.numel() + 1)
    ids = ids[:need].reshape(n_batches, batch, seq_len).to(device)
    return [ids[i] for i in range(n_batches)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--device", default="cpu")               # cpu: safe autograd thru the gate
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma

    print(f"=== gemma_sv stage-2 LoRA recovery: {args.model} on {args.device} ===")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
             .to(args.device))

    # Graft the SV gate onto the global layers, THEN inject LoRA into all attention
    # projections (the global gate's base.{q,k,v,o}_proj are matched by name too).
    graft_sv_into_gemma(model, nu=0.3, chunk=args.chunk, readout="softmax")
    lconf = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       task_type="CAUSAL_LM")
    model = get_peft_model(model, lconf)
    model.to(args.device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"LoRA rank {args.rank} on q/k/v/o_proj -- trainable {n_train:,} / {n_tot:,} "
          f"({100 * n_train / n_tot:.3f}%)")

    batches = build_lm_batches(tok, args.batch, args.seq_len, args.steps, args.device)
    opt = torch.optim.Adam(trainable, lr=args.lr)

    t0 = time.time()
    losses = []
    for step, ids in enumerate(batches):
        opt.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in trainable if p.grad is not None))
        opt.step()
        losses.append(out.loss.item())
        if step % max(1, args.steps // 8) == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  LM loss={out.loss.item():.4f}  |grad|={gnorm.item():.2e}")
    dt = time.time() - t0

    drop = 100 * (1 - losses[-1] / losses[0]) if losses[0] else 0.0
    print(f"\nLM loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({drop:.1f}% down) in {dt:.1f}s "
          f"({dt / args.steps:.2f}s/step)")
    assert losses[-1] < losses[0], "stage-2 LoRA did not reduce the LM loss"
    print("STAGE-2 SMOKE GREEN — LoRA on q/k/v/o_proj trains the grafted model, LM loss drops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

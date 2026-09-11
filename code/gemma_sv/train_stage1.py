"""Stage-1 attention transfer for the SV graft (LoLCATs-style); plan §step 4.

Distil the original global-attention behaviour into the grafted SV gate with the base
weights FROZEN -- only each global layer's RBF bandwidth (``kpar``) trains. This is the
SMOKE-scale wiring (tiny corpus, few steps, CPU) that proves the loop end to end:
  * ``kpar`` is a registered trainable ``nn.Parameter`` (one per global layer),
  * gradients reach it through the gate's implicit VJP, and
  * the attention-transfer MSE (grafted global output vs the original softmax output)
    drops -- the gate learns to stand in for softmax.

Scaling to the real recipe (~tens of M tokens, MPS/GPU) is a corpus + knob change; the
layer-wise-transfer fidelity upgrade and stage 2 (LoRA) are noted in distill.py. CPU is
the default because autograd through the batched gate solve is rock-solid there.

Run: .venv311/bin/python -m gemma_sv.train_stage1
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch

warnings.filterwarnings("ignore")

# Diverse public-domain-style prose; tiled to fill the (batch, seq_len) smoke batch.
CORPUS = (
    "The harbor town woke slowly under a pale grey sky. Fishing boats creaked against "
    "the dock as gulls argued over the morning catch. A clockmaker opened his shutters "
    "and wound the great brass mechanism in his window. Children chased a stray dog "
    "through the narrow cobbled lanes. Far to the north, the mountains held the last of "
    "the winter snow. A train whistle echoed across the valley and faded into the pines. "
    "In the library, a scholar copied figures from a worn astronomical table. The river "
    "carried small paper boats beneath the old stone bridge. By noon the market filled "
    "with the smell of bread, salt, and ripe summer fruit. An old sailor told the same "
    "story he always told, and the listeners forgave him for it. Lanterns were lit one "
    "by one as the evening tide came in. The town settled into a quiet that felt earned."
)


def build_batch(tok, batch: int, seq_len: int, device: str) -> dict:
    ids = tok(CORPUS, return_tensors="pt").input_ids[0]
    need = batch * seq_len
    if ids.numel() < need:                                   # tile to fill the batch
        ids = ids.repeat(need // ids.numel() + 1)
    ids = ids[:need].reshape(batch, seq_len).to(device)
    return {"input_ids": ids}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--device", default="cpu")               # cpu: safe autograd thru the gate solve
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.distill import (attention_transfer_loss,
                                  capture_global_reference_outputs, setup_stage1)
    from gemma_sv.layer_select import find_global_attention_layers

    print(f"=== gemma_sv stage-1 attention transfer: {args.model} on {args.device} ===")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
             .to(args.device).eval())
    batch = build_batch(tok, args.batch, args.seq_len, args.device)

    layer_ids = [i for i, _ in find_global_attention_layers(model)]
    print(f"global layers (teacher): {layer_ids}  | batch={args.batch} seq={args.seq_len} "
          f"chunk={args.chunk}")

    # Teacher targets: the ORIGINAL softmax global-attention outputs (capture pre-graft).
    refs = capture_global_reference_outputs(model, batch, layer_ids)

    # Graft, then freeze base + make each global layer's kpar trainable (init at median).
    graft_sv_into_gemma(model, nu=0.3, chunk=args.chunk, readout="softmax")
    params = setup_stage1(model, batch)
    model.to(args.device)                                    # keep new log_kpar on device
    n_train = sum(p.numel() for p in params)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train} / {n_tot:,} params ({100 * n_train / n_tot:.6f}%) -- per-layer kpar")

    glayers = dict(find_global_attention_layers(model))
    kpar0 = {i: glayers[i].self_attn.log_kpar.exp().item() for i in layer_ids}
    opt = torch.optim.Adam(params, lr=args.lr)

    t0 = time.time()
    losses = []
    for step in range(args.steps):
        opt.zero_grad()
        loss = attention_transfer_loss(model, batch, refs)
        loss.backward()
        gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 8) == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss={loss.item():.5e}  |grad|={gnorm.item():.2e}")
    dt = time.time() - t0

    kpar1 = {i: glayers[i].self_attn.log_kpar.exp().item() for i in layer_ids}
    print("\nkpar per global layer (init -> trained):")
    for i in layer_ids:
        print(f"  layer {i:2d}: {kpar0[i]:.3f} -> {kpar1[i]:.3f}")
    drop = 100 * (1 - losses[-1] / losses[0]) if losses[0] else 0.0
    print(f"loss {losses[0]:.4e} -> {losses[-1]:.4e}  ({drop:.1f}% down) in {dt:.1f}s "
          f"({dt / args.steps:.2f}s/step)")

    assert losses[-1] < losses[0], "stage-1 loss did not decrease"
    assert any(abs(kpar1[i] - kpar0[i]) > 1e-6 for i in layer_ids), \
        "kpar did not move -- no training signal reached the gate"
    print("STAGE-1 SMOKE GREEN — trainable kpar, gradient flow, decreasing attention-transfer loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

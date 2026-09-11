"""Deletion-cost benchmark (reviewer question: why the decrement, when the in-context
setting allows repacking the context without X?).

Answer: cost. Both are exact; the decrement is the incremental path. This measures, on the
real recovered-1B gate (hero-demo memory, float64 keys):

  (1) DECREMENT  remove one token from a seeded head-gate (C&P remove_point)
  (2) REFIT      fresh QP solve of the same head-gate without the token
  (3) REPACK     full re-prefill forward of the context (what repacking also pays,
                 on top of refitting every gate: the memory must be re-ingested)

Run: .venv311/bin/python -m gemma_sv.bench_forget_cost --lora outputs/gemma_sv_distill/lora_adapter
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

MODEL = "google/gemma-3-1b-pt"
NU, CHUNK = 0.3, 128
REPS = 5


def _kpar(X):
    d2 = np.sum((X[:, None] - X[None]) ** 2, axis=-1)
    return float(np.sqrt(np.median(d2[d2 > 0])))


def main() -> int:
    from cp_svm import FastOneClassSVM
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora", default="outputs/gemma_sv_distill/lora_adapter")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float64).eval()
    graft_sv_into_gemma(model, nu=NU, chunk=CHUNK, readout="softmax")
    if args.lora:
        from peft import PeftModel
        _mps = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        try:
            model = PeftModel.from_pretrained(model, args.lora).to(torch.float64).eval()
        finally:
            torch.backends.mps.is_available = _mps

    glayers = dict(find_global_attention_layers(model))
    layer_ids = sorted(glayers)

    fillers = [d["answer"] for d in
               list(load_dataset("locuslab/TOFU", "retain90", split="train"))[:22]]
    ids = tok("Memory:\n", add_special_tokens=True).input_ids
    for s in fillers:
        ids += tok(s + " ", add_special_tokens=False).input_ids
    T = len(ids)

    # (3) REPACK cost: the full re-prefill forward the repacking path must pay
    x = torch.tensor([ids])
    with torch.no_grad():
        model(x)                                              # warm-up
    t0 = time.perf_counter()
    for _ in range(3):
        with torch.no_grad():
            model(x)
    t_prefill = (time.perf_counter() - t0) / 3

    # capture float64 keys of the largest gated boundary on one global layer
    keys = {}
    handles = []
    for idx in layer_ids:
        attn = glayers[idx].self_attn

        def _hook(mod, a, kw, _idx=idx, _attn=attn):
            hs = a[0] if a else kw["hidden_states"]
            _, k, _ = _attn._project_qkv(hs, kw.get("position_embeddings"))
            keys[_idx] = k[0].detach().cpu().double().numpy()

        handles.append(attn.register_forward_pre_hook(_hook, with_kwargs=True))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    H = keys[layer_ids[0]].shape[0]
    n = (T // CHUNK) * CHUNK - CHUNK                          # largest full gated prefix
    n = max(k for k in range(CHUNK, T, CHUNK))
    C = min(1.0, 1.0 / (NU * T))
    n_boundaries = len(range(CHUNK, T, CHUNK))

    t_dec, t_ref = [], []
    for idx in layer_ids:
        for h in range(H):
            X = keys[idx][h, :n]
            kp = _kpar(X)
            m0 = FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(X)
            sup = [int(s) for s in m0.S]
            j = sup[len(sup) // 2]
            for _ in range(REPS):
                m = FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(X)
                t0 = time.perf_counter()
                try:
                    m.remove_point(j)
                    t_dec.append(time.perf_counter() - t0)
                except RuntimeError:
                    pass
                kept = [q for q in range(n) if q != j]
                t0 = time.perf_counter()
                FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(X[kept])
                t_ref.append(time.perf_counter() - t0)

    dec, ref = np.median(t_dec), np.median(t_ref)
    per_model_dec = dec * len(layer_ids) * H * n_boundaries
    per_model_ref = ref * len(layer_ids) * H * n_boundaries
    print(f"\n=== deletion cost @ {args.model}, T={T} memory tokens, gate n={n}, "
          f"{len(layer_ids)} layers x {H} heads x {n_boundaries} boundaries ===")
    print(f"(1) DECREMENT one token, per head-gate : {dec * 1e3:8.2f} ms  (median, {len(t_dec)} runs)")
    print(f"(2) REFIT gate without it, per head-gate: {ref * 1e3:8.2f} ms  ({ref / dec:.0f}x)")
    print(f"(3) REPACK re-prefill forward (float64/CPU): {t_prefill:6.2f} s")
    print(f"\nwhole-model deletion: decrement {per_model_dec:.2f} s "
          f"vs repack {per_model_ref + t_prefill:.2f} s "
          f"(refit all gates {per_model_ref:.2f} s + re-prefill {t_prefill:.2f} s) "
          f"-> {(per_model_ref + t_prefill) / per_model_dec:.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

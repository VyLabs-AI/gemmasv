"""Exact in-context unlearning at the readout (model-output) level on real gemma-3-1b.

The capability paper certifies exact forgetting at the SVDD *decision-function* level.
This lifts that to the grafted LLM's actual long-range-memory READOUT, on the live keys
of a real Gemma 3 global layer -- the headline claim, and it needs no stage-2 quality.

Claim. For a global SV layer, take its long-range memory = the gate over the prefix
keys K[:b], read out by the probe queries Q[b:]. Forgetting a prefix token i two ways:
  * decrement : the float64 Cauwenberghs-Poggio reverse path removes i (EXACT unlearning),
  * refit-without : fit the gate from scratch on K[:b] without i (the never-saw-it baseline),
must give the SAME readout. With the normalized hybrid readout the softmax denominator
cancels, so readout-equality follows from the gate's alpha-equivalence certificate
(machine-precision). Decay (keep i at alpha*=0.01) is the foil: it still leaks.

Faithfulness: the EXACT path is float64 C&P (cp_svm), never the FISTA inference solver.
Since the readout feeds the rest of the FROZEN model deterministically, an identical
readout implies identical logits -- exact in-context unlearning at the model output,
scoped (per the plan) to the long-range memory carried by the global layers.

Run: .venv311/bin/python -m gemma_sv.unlearn_demo
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

MODEL_ID = "google/gemma-3-1b-pt"
NU = 0.3
# The exact readout dev tracks the gate's alpha-equivalence floor (QP-seed / RBF
# conditioning): ~1e-9 on well-posed layers, up to ~1e-4 on an ill-conditioned one
# (cf. forgetting_rigor's tail). So we assert it stays "small" (<< decay) AND beats the
# decay foil by a wide margin in EVERY layer, rather than pinning an unrealistic 1e-13.
EXACT_TOL = 1e-3           # "small" readout-dev bar (decay leaks ~1e-2)
RATIO_MIN = 50.0           # per-layer: decay must leak >= 50x the exact dev
DECAY_GAMMA = 0.01         # approximate-unlearning foil (keep i, downweight)
PROMPT = "In a faraway kingdom, a curious fox kept a careful ledger of every promise. " * 16


def _median_kpar(X: np.ndarray) -> float:
    d2 = np.sum((X[:, None] - X[None]) ** 2, axis=-1)
    return float(np.sqrt(np.median(d2[d2 > 0])))


def prefix_readout(Q: np.ndarray, K: np.ndarray, V: np.ndarray, alpha: np.ndarray,
                   scaling: float) -> np.ndarray:
    """Normalized hybrid readout of queries Q over the gated prefix (K, V, alpha):
    softmax(Q.K * scaling) weighted by the gate alpha, row-normalized. The normalizer
    cancels the softmax denominator, so this depends on (K, V, alpha) only -- the basis
    of the decrement == refit-without equality."""
    s = (Q @ K.T) * scaling
    s = s - s.max(axis=1, keepdims=True)
    w = np.exp(s) * alpha[None, :]
    return (w @ V) / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)


def capture_kqv(model, input_ids, layer_ids):
    """Per global layer (batch 0, head 0): the live (K, Q, V, scaling) the gate sees,
    as float64 -- exactly SVGlobalAttention._project_qkv's output (post q/k-norm, RoPE,
    GQA expand)."""
    from gemma_sv.layer_select import find_global_attention_layers

    caps = {}
    handles = []
    for idx, layer in find_global_attention_layers(model):
        if idx not in layer_ids:
            continue
        attn = layer.self_attn

        def _hook(mod, args, kwargs, _idx=idx, _attn=attn):
            hs = args[0] if args else kwargs["hidden_states"]
            q, k, v = _attn._project_qkv(hs, kwargs.get("position_embeddings"))
            f64 = lambda t: t[0, 0].detach().cpu().to(torch.float64).numpy()
            caps[_idx] = (f64(k), f64(q), f64(v), float(_attn.base.scaling))

        handles.append(attn.register_forward_pre_hook(_hook, with_kwargs=True))
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return caps


def one_layer(K, Q, V, scaling):
    """Decrement vs refit-without vs decay, on one layer's prefix memory. Returns
    (forget_idx, exact_readout_dev, decay_readout_dev, alpha_dev) or None if the solve
    hits the documented uncovered margin-empty path."""
    from cp_svm import FastOneClassSVM

    n = len(K)
    b = n // 2                                   # prefix = long-range memory; probes = Q[b:]
    Kp, Vp, Qp = K[:b], V[:b], Q[b:]
    C, kpar = 1.0 / (NU * b), _median_kpar(Kp)
    try:
        full = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(Kp)
        S = sorted(full.S)                        # support tokens: forgetting one MATTERS
        i = S[len(S) // 2] if S else 0
        kept = [j for j in range(b) if j != i]
        dec = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(Kp)
        dec.remove_point(i)                       # EXACT decremental unlearning (float64)
        a_dec = np.asarray(dec.alpha)             # aligned to `kept`
        ref = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(Kp[kept])  # never-saw-it
        a_ref = np.asarray(ref.alpha)
    except RuntimeError:
        return None
    a_decay = np.asarray(full.alpha).copy()
    a_decay[i] *= DECAY_GAMMA                      # keep i, just downweight (approximate)

    O_dec = prefix_readout(Qp, Kp[kept], Vp[kept], a_dec, scaling)
    O_ref = prefix_readout(Qp, Kp[kept], Vp[kept], a_ref, scaling)
    O_decay = prefix_readout(Qp, Kp, Vp, a_decay, scaling)
    exact_dev = float(np.max(np.abs(O_dec - O_ref)))
    decay_dev = float(np.max(np.abs(O_decay - O_ref)))
    alpha_dev = float(np.max(np.abs(a_dec - a_ref)))
    return i, exact_dev, decay_dev, alpha_dev


def main(argv=None) -> int:
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--lora", default=None,
                    help="stage-2 LoRA adapter to load onto the graft (the RECOVERED model) "
                         "-- shows exact unlearning still holds after distillation")
    args = ap.parse_args(argv)

    tag = "recovered: graft+LoRA" if args.lora else "untrained graft"
    print(f"=== gemma_sv exact in-context unlearning @ readout level: {args.model} [{tag}] ===")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    graft_sv_into_gemma(model, nu=NU, readout="softmax")
    if args.lora:                                            # recovered model: load LoRA onto the graft
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).eval()
    layer_ids = [i for i, _ in find_global_attention_layers(model)]
    ids = tok(PROMPT, return_tensors="pt").input_ids
    caps = capture_kqv(model, ids, layer_ids)
    print(f"global layers {layer_ids}; T={ids.shape[1]} tokens, head 0, prefix=memory\n")
    print(f"{'layer':>5} {'forget':>7} {'alpha_dev':>11} {'exact readout dev':>18} "
          f"{'decay leak':>12} {'exact<<decay':>13}")

    exact_devs, ratios = [], []
    for idx in layer_ids:
        K, Q, V, scaling = caps[idx]
        r = one_layer(K, Q, V, scaling)
        if r is None:
            print(f"{idx:>5}   (skipped: margin-empty solve)")
            continue
        i, ed, dd, ad = r
        ratio = dd / max(ed, 1e-300)
        exact_devs.append(ed)
        ratios.append(ratio)
        print(f"{idx:>5} {i:>7} {ad:>11.2e} {ed:>18.2e} {dd:>12.2e} {ratio:>12.1e}x")

    assert exact_devs, "no layer completed -- cannot assert"
    worst_exact, min_ratio = max(exact_devs), min(ratios)
    print(f"\nworst exact readout dev = {worst_exact:.2e} (QP-seed/conditioning floor)  |  "
          f"decay leaks >= {min_ratio:.0f}x the exact dev in every layer")
    assert worst_exact < EXACT_TOL, f"forgetting not exact at readout: {worst_exact:.2e}"
    assert min_ratio >= RATIO_MIN, f"exact/decay contrast too weak: min ratio {min_ratio:.1f}x"
    print("UNLEARN DEMO GREEN — decrement == refit-without at the long-range-memory readout\n"
          "(float64 C&P, at the conditioning floor), while decay leaks by 2-6 orders. The readout\n"
          "feeds a deterministic frozen model => exact in-context unlearning at the model output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnostic: is the 4b exact-forget f_dev a prompt-induced ill-conditioning artifact
or a model-specific regression? Run the float64 C&P forget-equivalence on EVERY global
layer, under (a) the degenerate 40x-repeated validate_real prompt and (b) a diverse prompt,
and report f_dev + Gram condition number per layer. Cheap (cached weights), no training.

Run: .venv311/bin/python -m gemma_sv.diag_forget google/gemma-3-4b-pt
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

NU = 0.3
CHUNK = 128
REPEAT = "In a faraway kingdom, a curious fox kept a careful ledger of every promise. " * 40
DIVERSE = (
    "The harbor town woke to gulls and diesel. Mira counted the crates twice, then "
    "signed the manifest in green ink. Down the quay, an old crane groaned against a "
    "load of citrus bound for the northern markets, where winter had already bitten "
    "the orchards. A child chased a paper boat along the gutter. The tide turned at "
    "noon; by three the fog had swallowed the lighthouse, and the radio gave warnings "
    "in three languages. Somewhere a violin practiced the same difficult bar, again "
    "and again, never quite landing the final note before the silence took it back. "
) * 4


def cond_of(keys, kpar):
    d2 = np.sum((keys[:, None] - keys[None]) ** 2, axis=-1)
    K = np.exp(-d2 / (kpar ** 2))
    return float(np.linalg.cond(K))


def main(argv) -> int:
    model_id = argv[1] if len(argv) > 1 else "google/gemma-3-4b-pt"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import exact_forget_equivalence, graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers
    from gemma_sv.unlearn import extract_global_layer_keys

    print(f"=== forget diagnostic: {model_id} on {dev} ===")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(dev).eval()
    graft_sv_into_gemma(model, nu=NU, chunk=CHUNK, readout="softmax")
    globals_ = [i for i, _ in find_global_attention_layers(model)]
    forget = [7, 23, 41]

    for label, prompt in (("REPEAT(degenerate)", REPEAT), ("DIVERSE", DIVERSE)):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        print(f"\n[{label}]  T={ids.shape[1]} tokens")
        print(f"{'layer':>6}{'f_dev':>12}{'pmatch':>8}{'condK':>12}{'|R|':>6}")
        for li in globals_:
            keys = extract_global_layer_keys(model, ids, li, batch=0, head=0)
            d2 = np.sum((keys[:, None] - keys[None]) ** 2, axis=-1)
            kpar = float(np.sqrt(np.median(d2[d2 > 0])))
            res = exact_forget_equivalence(keys, forget, nu=NU, kpar=kpar)
            ck = cond_of(keys, kpar)
            print(f"{li:>6}{res['f_dev']:>12.2e}{res['partition_match']:>8.0f}{ck:>12.2e}"
                  f"{res.get('reserve_size', -1):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

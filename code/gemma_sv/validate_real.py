"""Real-weights validation of the SV graft on google/gemma-3-1b (plan step 3 / option B).

The same four checks as ``gemma_sv.smoke_test``, but against the ACTUAL pretrained
Gemma 3 1B instead of the random-init clone -- this is what confirms the graft, the
RoPE-on-keys / GQA / base-scaling plumbing, and the float64 exact-forget certificate
all work on real Gemma weights (not just a structurally-identical toy).

    1. summarize_layer_types   : the real interleave + global count
    2. graft_sv_into_gemma     : swap the global layers' self_attn for the SV gate
    3. one forward on real text: finite logits through the grafted globals
    4. exact_forget_equivalence: state-exact float64 forget on a real global layer's keys

GATED: needs Google's Gemma license accepted on the Hub + an HF token. Set it once:
    huggingface-cli login            # interactive (stores the token), or
    export HF_TOKEN=hf_xxx           # env var (read here)

Run:  .venv311/bin/python -m gemma_sv.validate_real
      .venv311/bin/python -m gemma_sv.validate_real google/gemma-3-1b-it   # variant
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

MODEL_ID = "google/gemma-3-1b-pt"     # base pretrained text model (-it = instruct variant)
NU = 0.3
CHUNK = 128                            # scaffold default; T must exceed it to fire the gate
EXACT_TOL = 1e-5                       # machine-precision-ish bar (see smoke_test rationale)
# Long enough (~360 tokens) that gated chunk boundaries land inside the sequence.
PROMPT = "In a faraway kingdom, a curious fox kept a careful ledger of every promise. " * 40


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_real_gemma(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")  # None -> fall back to cached huggingface-cli login
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token, dtype=torch.float32)
    return model.eval(), tok


def main(argv) -> int:
    model_id = argv[1] if len(argv) > 1 else MODEL_ID
    device = pick_device()
    print(f"=== gemma_sv real-weights validation: {model_id} on {device} ===")

    try:
        model, tok = load_real_gemma(model_id)
    except Exception as e:                                    # gated / no-token / offline
        print(f"\nFAILED to load the gated model: {type(e).__name__}: {e}\n"
              f"-> Accept the license at https://huggingface.co/{model_id} , then set a\n"
              f"   token: `huggingface-cli login`  or  `export HF_TOKEN=hf_...`  and re-run.")
        return 2
    model = model.to(device)

    from gemma_sv import (exact_forget_equivalence, graft_sv_into_gemma,
                          summarize_layer_types)
    from gemma_sv.layer_select import find_global_attention_layers
    from gemma_sv.sv_global_attention import SVGlobalAttention
    from gemma_sv.unlearn import certified_reserve, extract_global_layer_keys

    # [1] real interleave
    print(f"[1] {summarize_layer_types(model)}")
    globals_ = [i for i, _ in find_global_attention_layers(model)]
    print(f"    global layer indices: {globals_}")
    assert globals_, "no global (non-sliding) layers found on the real model"

    # [2] graft the SV gate onto the real global layers
    replaced = graft_sv_into_gemma(model, nu=NU, chunk=CHUNK, readout="softmax")
    post = [i for i, _ in find_global_attention_layers(model)]
    print(f"[2] grafted SV gate into {len(replaced)} global layers {replaced}")
    assert replaced == globals_ and post == globals_, "grafted globals not rediscoverable"
    for idx, layer in find_global_attention_layers(model):
        assert isinstance(layer.self_attn, SVGlobalAttention)

    # [3] one forward on real tokenized text through the grafted globals
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    print(f"[3] forward ok: T={ids.shape[1]} tokens, logits {tuple(logits.shape)}, "
          f"finite={bool(torch.isfinite(logits).all())}")
    assert torch.isfinite(logits).all(), "non-finite logits from grafted real forward"
    assert ids.shape[1] > CHUNK, f"prompt too short ({ids.shape[1]}<=CHUNK={CHUNK}); gate idle"

    # [4] exact forgetting on a real global layer's live keys
    layer_idx = globals_[0]
    keys = extract_global_layer_keys(model, ids, layer_idx, batch=0, head=0)
    d2 = np.sum((keys[:, None] - keys[None]) ** 2, axis=-1)
    kpar = float(np.sqrt(np.median(d2[d2 > 0])))
    forget = [7, 23, 41]
    res = exact_forget_equivalence(keys, forget, nu=NU, kpar=kpar)
    reserve = certified_reserve(keys, nu=NU, kpar=kpar)
    pm = res["partition_match"]
    print(f"[4] global layer {layer_idx}: keys {keys.shape}, kpar={kpar:.3g}, forget={forget}")
    print(f"    exact-forget f_dev={res['f_dev']:.2e}  partition_match={pm:.0f}"
          f"  C={res['C']:.4g}  reserve(|R|)={len(reserve)}")
    # The reviewer-proof exactness claim is FUNCTIONAL (decision-function deviation),
    # per forgetting_rigor: it is invariant to the non-unique optimum. With a large key
    # set and the tiny box C=1/(nu*n), many points pin at the bound, so the (S,E,R)
    # partition of the C&P decrement vs a from-scratch refit can legitimately differ
    # even though the gate is the SAME function -- so partition_match is reported, not
    # asserted (it is exact only when the optimum is unique, e.g. the smoke test).
    assert res["f_dev"] < EXACT_TOL, \
        f"forget not functionally exact on real keys: f_dev={res['f_dev']:.2e} >= {EXACT_TOL:.0e}"
    if pm < 1.0:
        print("    (partition_match<1 expected at this n/C: non-unique optimum; functional "
              "exactness is the claim and it holds -- cf. forgetting_rigor.)")

    print(f"\nREAL-WEIGHTS VALIDATION GREEN — the SV graft runs on {model_id}: real 5:1 "
          f"interleave, finite forward, functionally-exact forgetting (f_dev={res['f_dev']:.1e}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

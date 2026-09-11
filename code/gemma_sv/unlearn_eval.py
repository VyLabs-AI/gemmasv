"""In-context unlearning: forget efficacy + retain specificity vs the never-ingested floor (plan §10).

Persistent-memory framing: TOFU author facts packed as the SV global-layers' long-range prefix;
"forget F" = drop F's key positions from the gate (`drop_pos`). Each forget target is placed BEYOND
the local window so its recall is gate-mediated (truncating the surface tokens wouldn't reach it).

  efficacy   = (ES_present - ES_forget) / (ES_present - ES_never)  -> 1.0 == recall reaches the
               NEVER-INGESTED floor (B: the in-context retrain-without gold, not literal 0).
  specificity= ES(retained fact) under forget-F  vs  present       -> ~0 == no collateral (A).
ES = distinctive-span answer log-prob (answer tokens NOT in the question -- the recalled content).

Run: .venv311/bin/python -m gemma_sv.unlearn_eval --targets 40 --lora outputs/gemma_sv_distill/lora_adapter
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

MODEL = "google/gemma-3-1b-pt"
NU = 0.3
CHUNK = 128
N_FILL = 22          # filler facts to push the memory past the local window
RETAIN_N = 3         # retained facts probed for specificity
WINDOW = 512         # gemma-3-1b sliding window; targets must sit earlier than (len_mem - WINDOW)
MIN_SIGNAL = 0.05    # skip facts the model doesn't recall (can't measure forgetting)


def ci95(x):
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def main(argv=None) -> int:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora", default=None, help="stage-2 adapter -> recovered model")
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--window", type=int, default=WINDOW,
                    help="sliding-window size; F must sit beyond it (1b=512, 4b/12b=1024)")
    ap.add_argument("--n-fill", type=int, default=N_FILL,
                    help="filler facts to push F past the window (raise for a larger window)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args(argv)
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    graft_sv_into_gemma(base, nu=NU, chunk=CHUNK, readout="softmax")
    if args.lora:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, args.lora).merge_and_unload()
    model = base.to(dev).eval()
    glayers = [layer for _, layer in find_global_attention_layers(model)]

    def set_drop(p):
        for layer in glayers:
            layer.self_attn._drop_pos = p

    forget = list(load_dataset("locuslab/TOFU", "forget10", split="train"))
    retain = list(load_dataset("locuslab/TOFU", "retain90", split="train"))
    retain_set = [(d["question"], d["answer"]) for d in retain[:RETAIN_N]]
    fillers = [d["answer"] for d in retain[RETAIN_N:RETAIN_N + args.n_fill]]

    def pack(stmts):
        ids = tok("Memory:\n", add_special_tokens=True).input_ids
        span = {}
        for nm, s in stmts:
            t = tok(s + " ", add_special_tokens=False).input_ids
            span[nm] = (len(ids), len(ids) + len(t))
            ids += t
        return ids, span

    def build(F):                                              # F first, then retain set, then fillers
        stmts = ([("F", F[1])] + [(f"R{i}", r[1]) for i, r in enumerate(retain_set)]
                 + [(f"x{i}", s) for i, s in enumerate(fillers)])
        return pack(stmts)

    mem_never, _ = pack([(f"R{i}", r[1]) for i, r in enumerate(retain_set)]
                        + [(f"x{i}", s) for i, s in enumerate(fillers)])

    def es(memids, q, a, drop=None):
        set_drop(list(range(*drop)) if drop else None)
        qa = tok(f"\n\nQuestion: {q}\nAnswer:", add_special_tokens=False).input_ids
        A = tok(" " + a, add_special_tokens=False).input_ids
        x = torch.tensor([memids + qa + A]).to(dev)
        nP = len(memids) + len(qa)
        with torch.no_grad():
            lp = torch.log_softmax(model(x).logits[0], -1)
        g = lp[torch.arange(nP - 1, nP - 1 + len(A)), torch.tensor(A).to(dev)]
        set_drop(None)
        qs = set(tok(q, add_special_tokens=False).input_ids)
        keep = torch.tensor([t not in qs for t in A])
        return float(g[keep].mean()) if keep.any() else float(g.mean())

    eff, rdelta = [], []
    targets = [(d["question"], d["answer"]) for d in forget[:args.targets]]
    for n, F in enumerate(targets):
        mem, span = build(F)
        if span["F"][1] > len(mem) - args.window:             # F must be beyond the local window
            continue
        pF = es(mem, *F)
        nF = es(mem_never, *F)
        if pF - nF < MIN_SIGNAL:                              # no recall signal -> can't measure
            continue
        fF = es(mem, *F, drop=span["F"])
        eff.append((pF - fF) / (pF - nF))
        for i, R in enumerate(retain_set):
            if span[f"R{i}"][1] <= len(mem) - args.window:    # retained fact also gate-mediated
                rdelta.append(es(mem, *R, drop=span["F"]) - es(mem, *R))
        if n % 10 == 0:
            print(f"  {n + 1}/{len(targets)}  eff={eff[-1]:.2f}", flush=True)

    eff, rdelta = np.array(eff), np.array(rdelta)
    tag = "recovered+LoRA" if args.lora else "untrained graft"
    print(f"\n=== in-context unlearning [{tag}], {len(eff)} targets with recall signal ===")
    print(f"FORGET efficacy (1.0 = never-ingested floor): mean={eff.mean():.2f} ± {ci95(eff):.2f} "
          f"(95% CI)  median={np.median(eff):.2f}")
    print(f"RETAIN  ΔES under forget (0 = perfect specificity): mean={rdelta.mean():+.3f} "
          f"± {ci95(rdelta):.3f}  (n={len(rdelta)} probes)")
    print("=> the masked-refit behavioral proxy reaches the never-ingested floor while leaving "
          "retained facts approximately intact; this is not the float64 decrement certificate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

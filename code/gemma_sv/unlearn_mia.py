"""LiRA-style membership-inference on in-context unlearning (plan §10, eval C).

After forgetting F, can an attacker still tell F was ingested? We vary the memory context (filler
subsets = "shadow" draws) and pool the extraction score ES(F) under each condition, then compute
AUC distinguishing "F-ingested-then-forgotten" from "F never ingested" (the in-context analog of
LiRA-Forget; no shadow-model training needed -- shadow CONTEXTS suffice):

  AUC(condition vs never)  ->  0.5 = membership ERASED (forgotten == never-ingested)
                              1.0 = membership fully detectable (F still there)

Expect: present ~1.0 (F obviously detectable), SV-exact ~0.5 (erased), ICUL ~1.0 (never forgot),
decay ~0.5 (heavy decay; its leak is on the fine metrics, not coarse membership).

Run: .venv311/bin/python -m gemma_sv.unlearn_mia --targets 40 --configs 3 --lora outputs/gemma_sv_distill/lora_adapter
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

MODEL = "google/gemma-3-1b-pt"
NU, CHUNK, N_FILL, WINDOW, MIN_SIGNAL, DECAY = 0.3, 128, 22, 512, 0.05, 0.01
ICUL_INSTR = ("Instruction: the information required by the FINAL question has been permanently "
              "deleted from memory and must not be answered.\n\n")
CONDS = ["present", "sv_exact", "decay", "icul"]


def main() -> int:
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--configs", type=int, default=3)
    ap.add_argument("--window", type=int, default=WINDOW,
                    help="sliding-window size; F must sit beyond it (1b=512, 4b/12b=1024)")
    ap.add_argument("--n-fill", type=int, default=N_FILL,
                    help="filler facts per shadow draw (raise for a larger window)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    n_targets = 4 if args.smoke else args.targets
    n_cfg = 2 if args.smoke else args.configs

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    graft_sv_into_gemma(base, nu=NU, chunk=CHUNK, readout="softmax")
    if args.lora:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, args.lora).merge_and_unload()
    model = base.to(dev).eval()
    glayers = [layer for _, layer in find_global_attention_layers(model)]

    def setfx(drop=None, scale=None):
        for layer in glayers:
            layer.self_attn._drop_pos = drop
            layer.self_attn._scale_pos = scale
            layer.self_attn._scale_factor = DECAY if scale else 1.0

    forget = list(load_dataset("locuslab/TOFU", "forget10", split="train"))
    retain = [d["answer"] for d in load_dataset("locuslab/TOFU", "retain90", split="train")]

    def pack(stmts):
        ids = tok("Memory:\n", add_special_tokens=True).input_ids
        span = {}
        for nm, s in stmts:
            t = tok(s + " ", add_special_tokens=False).input_ids
            span[nm] = (len(ids), len(ids) + len(t))
            ids += t
        return ids, span

    def es(memids, suffix, q, a, drop=None, scale=None):
        setfx(drop, scale)
        suf = tok(suffix, add_special_tokens=False).input_ids
        A = tok(" " + a, add_special_tokens=False).input_ids
        x = torch.tensor([memids + suf + A]).to(dev)
        nP = len(memids) + len(suf)
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=x).logits[0], -1)
        g = lp[torch.arange(nP - 1, nP - 1 + len(A)), torch.tensor(A).to(dev)]
        setfx(None, None)
        qs = set(tok(q, add_special_tokens=False).input_ids)
        keep = torch.tensor([t not in qs for t in A])
        return float(g[keep].mean()) if keep.any() else float(g.mean())

    scores = {c: [] for c in CONDS}
    never = []
    targets = [(d["question"], d["answer"]) for d in forget[:n_targets]]
    for n, F in enumerate(targets):
        for j in range(n_cfg):                                       # shadow draw = filler subset
            fl = retain[j * args.n_fill:(j + 1) * args.n_fill]
            if len(fl) < args.n_fill:
                break
            mem, span = pack([("F", F[1])] + [(f"x{i}", s) for i, s in enumerate(fl)])
            mem_never, _ = pack([(f"x{i}", s) for i, s in enumerate(fl)])
            if span["F"][1] > len(mem) - args.window:
                continue
            qb = f"\n\nQuestion: {F[0]}\nAnswer:"
            nv = es(mem_never, qb, *F)
            pr = es(mem, qb, *F)
            if pr - nv < MIN_SIGNAL:
                continue
            fspan = list(range(*span["F"]))
            never.append(nv)
            scores["present"].append(pr)
            scores["sv_exact"].append(es(mem, qb, *F, drop=fspan))
            scores["decay"].append(es(mem, qb, *F, scale=fspan))
            scores["icul"].append(es(mem, f"\n\n{ICUL_INSTR}Question: {F[0]}\nAnswer:", *F))
        if n % 10 == 0:
            print(f"  {n + 1}/{len(targets)} (samples {len(never)})", flush=True)

    y = [1] * len(never) + [0] * len(never)
    tag = "recovered+LoRA" if args.lora else "untrained graft"
    print(f"\n=== LiRA-style MIA [{tag}], {len(never)} (target,config) draws ===")
    print(f"{'condition':<12}{'AUC(vs never)':>14}   interpretation")
    notes = {"present": "F present (sanity: detectable)", "sv_exact": "EXACT: membership erased→0.5",
             "decay": "decay (coarse; fine leak elsewhere)", "icul": "ICUL: never forgot→high"}
    for c in CONDS:
        auc = roc_auc_score(y, scores[c] + never)
        print(f"{c:<12}{auc:>14.3f}   {notes[c]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

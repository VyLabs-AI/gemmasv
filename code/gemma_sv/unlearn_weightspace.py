"""Weight-space unlearning baseline, run in the in-context framing (reviewer ask: one
representative per family -- prompt (ICUL), in-mechanism approximate (decay), in-mechanism
exact (ours), and WEIGHT-SPACE (this script)).

Per target F (packed beyond the local window, as everywhere in the paper): fine-tune a fresh
LoRA by GRADIENT ASCENT on F's answer tokens given (memory + question) -- the standard GA
unlearner, budget-matched by early-stopping the moment recall reaches the never-ingested
floor (the same stopping rule that tuned decay's gamma). Then measure, on the adapted model:

  residual   = (ES_F - floor) / (ES_F_pre - floor_pre)   0 = at the floor, 1 = no forgetting
  collateral = mean ΔES on retain probes (weight edits hit the shared read-from-memory
               mechanism; expect < 0 where exact deletion gives ~0)
  recovery   = residual after a benign relearning fine-tune (PrivUn protocol, m samples;
               expect it to climb back -- suppression, not removal)

The structural point this baseline makes: weight editing cannot remove a fact that lives in
the CONTEXT; it can only damage the model's ability to read it -- which is collateral,
reversible, and carries no certificate.

Run: .venv311/bin/python -m gemma_sv.unlearn_weightspace --lora outputs/gemma_sv_distill/lora_adapter
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

MODEL = "google/gemma-3-1b-pt"
NU, CHUNK, N_FILL, WINDOW, MIN_SIGNAL = 0.3, 128, 22, 512, 0.05
RETAIN_N = 3
GA_LR = 1e-4
GA_MAX_STEPS = 60
GA_CHECK_EVERY = 5
RELEARN_M = 64
RELEARN_LR = 1e-3


def ci95(x):
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main() -> int:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lora", default=None, help="stage-2 adapter -> recovered model")
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--relearn-targets", type=int, default=10)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--n-fill", type=int, default=N_FILL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="outputs/gemma_sv_eval/weightspace.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    n_targets = 2 if args.smoke else args.targets
    n_relearn = 1 if args.smoke else args.relearn_targets
    relearn_m = 8 if args.smoke else RELEARN_M

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    graft_sv_into_gemma(base, nu=NU, chunk=CHUNK, readout="softmax")
    if args.lora:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, args.lora).merge_and_unload()
    base = base.to(dev).eval()
    glayers = [layer for _, layer in find_global_attention_layers(base)]

    forget = list(load_dataset("locuslab/TOFU", "forget10", split="train"))
    retain = list(load_dataset("locuslab/TOFU", "retain90", split="train"))
    retain_set = [(d["question"], d["answer"]) for d in retain[:RETAIN_N]]
    fillers = [d["answer"] for d in retain[RETAIN_N:RETAIN_N + args.n_fill]]
    relearn_pool = [(d["question"], d["answer"])
                    for d in retain[RETAIN_N + args.n_fill:RETAIN_N + args.n_fill + 300]]

    def pack(stmts):
        ids = tok("Memory:\n", add_special_tokens=True).input_ids
        span = {}
        for nm, s in stmts:
            t = tok(s + " ", add_special_tokens=False).input_ids
            span[nm] = (len(ids), len(ids) + len(t))
            ids += t
        return ids, span

    def build(F):
        return pack([("F", F[1])] + [(f"R{i}", r[1]) for i, r in enumerate(retain_set)]
                    + [(f"x{i}", s) for i, s in enumerate(fillers)])

    mem_never, _ = pack([(f"R{i}", r[1]) for i, r in enumerate(retain_set)]
                        + [(f"x{i}", s) for i, s in enumerate(fillers)])

    def es(model, memids, q, a):
        qa = tok(f"\n\nQuestion: {q}\nAnswer:", add_special_tokens=False).input_ids
        A = tok(" " + a, add_special_tokens=False).input_ids
        x = torch.tensor([memids + qa + A]).to(dev)
        nP = len(memids) + len(qa)
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=x).logits[0], -1)
        g = lp[torch.arange(nP - 1, nP - 1 + len(A)), torch.tensor(A).to(dev)]
        qs = set(tok(q, add_special_tokens=False).input_ids)
        keep = torch.tensor([t not in qs for t in A])
        return float(g[keep].mean()) if keep.any() else float(g.mean())

    def ga_batch(F, memids):
        """(input_ids, labels) for ascent on F's answer tokens given memory+question."""
        qa = tok(f"\n\nQuestion: {F[0]}\nAnswer:", add_special_tokens=False).input_ids
        A = tok(" " + F[1], add_special_tokens=False).input_ids
        ids = memids + qa + A
        labels = [-100] * (len(memids) + len(qa)) + A
        return (torch.tensor([ids]).to(dev), torch.tensor([labels]).to(dev))

    res = {"residual": [], "collateral": [], "recovery": [], "ga_steps": []}
    targets = [(d["question"], d["answer"]) for d in forget[:n_targets]]
    for n, F in enumerate(targets):
        mem, span = build(F)
        if span["F"][1] > len(mem) - args.window:
            continue
        pre = es(base, mem, *F)
        floor_pre = es(base, mem_never, *F)
        if pre - floor_pre < MIN_SIGNAL:
            continue
        r_pre = [es(base, mem, *R) for R in retain_set]

        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        pm = get_peft_model(base, cfg).to(dev).train()
        opt = torch.optim.Adam([p for p in pm.parameters() if p.requires_grad], lr=GA_LR)
        x, lab = ga_batch(F, mem)
        steps = GA_MAX_STEPS
        for i in range(GA_MAX_STEPS):
            opt.zero_grad()
            (-pm(input_ids=x, labels=lab).loss).backward()     # gradient ASCENT
            torch.nn.utils.clip_grad_norm_(
                [p for p in pm.parameters() if p.requires_grad], 1.0)
            opt.step()
            if (i + 1) % GA_CHECK_EVERY == 0:
                pm.eval()
                if es(pm, mem, *F) <= es(pm, mem_never, *F):   # reached the (shifted) floor
                    steps = i + 1
                    pm.train()
                    break
                pm.train()
        pm.eval()

        after = es(pm, mem, *F)
        floor_after = es(pm, mem_never, *F)
        residual = (after - floor_after) / (pre - floor_pre)
        collateral = float(np.mean([es(pm, mem, *R) - r for R, r in zip(retain_set, r_pre)]))
        res["residual"].append(residual)
        res["collateral"].append(collateral)
        res["ga_steps"].append(steps)

        if len(res["recovery"]) < n_relearn:                   # PrivUn relearning attack
            opt2 = torch.optim.Adam([p for p in pm.parameters() if p.requires_grad],
                                    lr=RELEARN_LR)
            pm.train()
            for i in range(relearn_m):
                blk = relearn_pool[(i * 6) % (len(relearn_pool) - 6):][:6]
                ctx = "Memory:\n" + "".join(a + " " for _, a in blk)
                text = ctx + f"\n\nQuestion: {blk[0][0]}\nAnswer: {blk[0][1]}"
                ids = tok(text, return_tensors="pt").input_ids.to(dev)
                opt2.zero_grad()
                pm(input_ids=ids, labels=ids).loss.backward()
                opt2.step()
            pm.eval()
            rec = (es(pm, mem, *F) - es(pm, mem_never, *F)) / (pre - floor_pre)
            res["recovery"].append(rec)

        base = pm.unload()                                     # restore clean weights
        base.eval()
        glayers = [layer for _, layer in find_global_attention_layers(base)]
        if dev == "mps":
            torch.mps.empty_cache()
        print(f"  {n + 1}/{len(targets)}  steps={steps}  residual={residual:+.2f}  "
              f"collateral={collateral:+.3f}"
              + (f"  recovery={res['recovery'][-1]:+.2f}" if len(res['recovery']) and
                 len(res["residual"]) == len(res["recovery"]) else ""), flush=True)

    tag = "recovered+LoRA" if args.lora else "untrained graft"
    print(f"\n=== weight-space GA baseline in the in-context framing [{tag}], "
          f"{len(res['residual'])} targets ===")
    print(f"GA budget: {np.mean(res['ga_steps']):.0f} steps mean (cap {GA_MAX_STEPS}), "
          f"early-stopped at the never-ingested floor")
    print(f"residual signal after GA : {np.mean(res['residual']):+.2f} ± "
          f"{ci95(res['residual']):.2f}   (0 = floor)")
    print(f"retain collateral ΔES    : {np.mean(res['collateral']):+.3f} ± "
          f"{ci95(res['collateral']):.3f} (exact deletion: ~0)")
    print(f"relearning recovery (m={RELEARN_M}): {np.mean(res['recovery']):+.2f} ± "
          f"{ci95(res['recovery']):.2f}   (n={len(res['recovery'])}; exact stays at floor)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {k: v for k, v in res.items()} | {"config": vars(args)}, indent=2, default=str))
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

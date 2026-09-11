"""FULL LiRA membership inference on in-context unlearning (per-example likelihood ratios).

Upgrades the first-cut pooled-score test (unlearn_mia) to the full Carlini et al. protocol,
in its faithful in-context form: ingestion here is PACKING the record into the context
memory, not gradient training, so the shadow ensemble is shadow CONTEXTS -- for each target
F, N random filler packings WITH F ("in") and WITHOUT F ("out"). We fit per-target Gaussians
to the extraction score under each world, and the attack statistic on a test observation s is

  Lambda(s) = log N(s; mu_in, sig_in) - log N(s; mu_out, sig_out).

Held-out test draws give positives ("ingested then unlearned by method M") and negatives
("never ingested"); we report AUC and TPR at low FPR (the metric LiRA insists on).
0.5 AUC / TPR==FPR = membership ERASED. Shadow-model TRAINING is deliberately absent:
training is not the ingestion mechanism in-context, so trained shadows test the wrong channel.

Run: .venv311/bin/python -m gemma_sv.unlearn_mia_lira --lora outputs/gemma_sv_distill/lora_adapter
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

MODEL = "google/gemma-3-1b-pt"
NU, CHUNK, N_FILL, WINDOW, MIN_SIGNAL, DECAY = 0.3, 128, 22, 512, 0.05, 0.01
ICUL_INSTR = ("Instruction: the information required by the FINAL question has been permanently "
              "deleted from memory and must not be answered.\n\n")
CONDS = ["present", "sv_exact", "decay", "icul"]
SIG_FLOOR = 1e-2      # variance floor for the per-target Gaussians (few-shadow stability)


def tpr_at_fpr(y, s, fpr_grid=(0.01, 0.05, 0.10)):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return {f: float(np.interp(f, fpr, tpr)) for f in fpr_grid}


def main() -> int:
    from datasets import load_dataset
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-revision")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--indices", help="comma-separated forget10 row indices")
    ap.add_argument("--shadows", type=int, default=32, help="shadow draws per side (Gaussian fit)")
    ap.add_argument("--tests", type=int, default=8, help="held-out test draws per target")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--n-fill", type=int, default=N_FILL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nu", type=float, default=NU)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--solver-seed", type=int, default=0)
    ap.add_argument("--preserve-prefix-mass", action="store_true")
    ap.add_argument("--per-boundary-box", action="store_true")
    ap.add_argument("--out", default="outputs/gemma_sv_eval/mia_lira.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    n_targets = 3 if args.smoke else args.targets
    n_shadow = 6 if args.smoke else args.shadows
    n_test = 2 if args.smoke else args.tests
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=torch.float32,
    )
    graft_sv_into_gemma(
        base,
        nu=args.nu,
        chunk=args.chunk,
        readout="softmax",
        preserve_prefix_mass=args.preserve_prefix_mass,
        per_boundary_box=args.per_boundary_box,
        solver_seed=args.solver_seed,
    )
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
    pool = [d["answer"] for d in load_dataset("locuslab/TOFU", "retain90", split="train")]

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

    def draw(F, with_f):
        """One shadow/test packing: random filler subset (+F first when with_f)."""
        fl = rng.sample(pool, args.n_fill)
        stmts = ([("F", F[1])] if with_f else []) + [(f"x{i}", s) for i, s in enumerate(fl)]
        mem, span = pack(stmts)
        return mem, (span.get("F"), len(mem))

    lam = {c: [] for c in CONDS}          # positives per condition (Lambda values)
    lam_never = []                        # shared negatives
    used = 0
    target_indices = (
        [int(value) for value in args.indices.split(",") if value.strip()]
        if args.indices
        else list(range(n_targets))
    )
    targets = [
        (forget[index]["question"], forget[index]["answer"])
        for index in target_indices
    ]
    for n, F in enumerate(targets):
        qb = f"\n\nQuestion: {F[0]}\nAnswer:"
        s_in, s_out = [], []
        for _ in range(n_shadow):                              # shadow ensemble (Gaussian fit)
            mem, (sp, L) = draw(F, True)
            if sp[1] > L - args.window:
                continue
            s_in.append(es(mem, qb, *F))
            mem_o, _ = draw(F, False)
            s_out.append(es(mem_o, qb, *F))
        if len(s_in) < 4:
            continue
        mu_i, sd_i = np.mean(s_in), max(np.std(s_in, ddof=1), SIG_FLOOR)
        mu_o, sd_o = np.mean(s_out), max(np.std(s_out, ddof=1), SIG_FLOOR)
        if mu_i - mu_o < MIN_SIGNAL:                           # no recall signal -> unmeasurable
            continue
        used += 1

        def L(s):
            return float(norm.logpdf(s, mu_i, sd_i) - norm.logpdf(s, mu_o, sd_o))

        for _ in range(n_test):                                # held-out test draws
            mem, (sp, Lm) = draw(F, True)
            if sp[1] > Lm - args.window:
                continue
            fspan = list(range(*sp))
            lam["present"].append(L(es(mem, qb, *F)))
            lam["sv_exact"].append(L(es(mem, qb, *F, drop=fspan)))
            lam["decay"].append(L(es(mem, qb, *F, scale=fspan)))
            lam["icul"].append(L(es(mem, f"\n\n{ICUL_INSTR}Question: {F[0]}\nAnswer:", *F)))
            mem_o, _ = draw(F, False)
            lam_never.append(L(es(mem_o, qb, *F)))
        if n % 5 == 0:
            print(f"  {n + 1}/{len(targets)} targets (kept {used}, tests {len(lam_never)})",
                  flush=True)

    tag = "recovered+LoRA" if args.lora else "untrained graft"
    print(f"\n=== FULL LiRA (shadow contexts, per-target Gaussians, likelihood ratio) [{tag}] ===")
    print(f"{used} measurable targets x {n_test} held-out draws = {len(lam_never)} test points/side;"
          f" {n_shadow} shadows/side")
    print(f"{'condition':<12}{'AUC':>8}{'TPR@1%':>9}{'TPR@5%':>9}{'TPR@10%':>9}   interpretation")
    notes = {"present": "F present (sanity: detectable)",
             "sv_exact": "masked refit (behavioral proxy)",
             "decay": "decay (coarse membership)",
             "icul": "ICUL: never forgot -> high"}
    report = {
        "schema": "gemma-sv-lira-v2",
        "config": vars(args),
        "target_indices": target_indices,
        "n_targets_used": used,
        "n_tests": len(lam_never),
    }
    y = [1] * len(lam_never) + [0] * len(lam_never)
    for c in CONDS:
        s = lam[c] + lam_never
        auc = roc_auc_score(y, s)
        t = tpr_at_fpr(y, s)
        print(f"{c:<12}{auc:>8.3f}{t[0.01]:>9.3f}{t[0.05]:>9.3f}{t[0.10]:>9.3f}   {notes[c]}")
        report[c] = {"auc": float(auc), "tpr_at_fpr": {str(k): v for k, v in t.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

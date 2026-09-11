"""Scaled SV-graft distillation on real data, with held-out WikiText-103 perplexity.

End-to-end, launchable, checkpointed (plan §step 5):
  ppl(original) -> graft -> stage-1 attention transfer -> stage-2 LoRA recovery
  -> ppl(grafted+recovered), all on FineWeb-Edu, eval on WikiText-103 (quality parity).

Tuned for the M3 Ultra (MPS, big unified memory -> push --batch up to amortise the gate's
FISTA). Throughput ~1.9k tok/s at B=4 on gemma-3-1b => ~6 h / 40M tokens. token budget =
(stage1-steps + stage2-steps) * batch * seq-len.

  # quick pipeline check (~2 min):
  .venv311/bin/python -m gemma_sv.run_distill --stage1-steps 10 --stage2-steps 20 --eval-blocks 20
  # a real run (~20M tokens, a few hours), checkpointing under outputs/:
  .venv311/bin/python -m gemma_sv.run_distill --batch 8 --stage1-steps 2000 --stage2-steps 8000

Rigorous parity needs the CONTROL too (original + identical LoRA budget, no graft) to isolate
the gate's cost from plain fine-tuning -- run with --control (doubles stage-2 time).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Stage-1's bandwidth gradient includes cdist backward, which PyTorch does not
# implement natively on MPS. This must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from gemma_sv.recovery_protocol import (
    BatchFingerprint,
    FINEWEB_REVISION,
    MODEL_REVISION,
    WIKITEXT_REVISION,
    path_sha256,
    seed_everything,
    tensor_sha256,
    validate_stage2_pair,
    write_json_atomic,
)
from gemma_sv.recovery_state import collect_recovery_state, save_recovery_state


def _load(model_id, device, revision):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
    ).to(device)


def _lora(model, rank, device):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg).to(device)


def _train(
    model,
    opt,
    data_iter,
    n_steps,
    loss_fn,
    label,
    log_every,
    *,
    fingerprint=None,
):
    t0, last = time.time(), 0.0
    for step in range(n_steps):
        ids = next(data_iter)
        if fingerprint is not None:
            fingerprint.update(ids)
        opt.zero_grad()
        loss = loss_fn(ids)
        loss.backward()
        opt.step()
        last = loss.item()
        if step % log_every == 0 or step == n_steps - 1:
            tok = (step + 1) * ids.numel()
            print(f"  [{label}] step {step:5d}/{n_steps}  loss={last:.4f}  "
                  f"{tok / (time.time() - t0):.0f} tok/s", flush=True)
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument("--fineweb-revision", default=FINEWEB_REVISION)
    ap.add_argument("--wikitext-revision", default=WIKITEXT_REVISION)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument(
        "--preserve-prefix-mass",
        action="store_true",
        help=(
            "rescale the gated long-range prefix to its pre-gate softmax "
            "mass while keeping reserve coefficients at exact zero"
        ),
    )
    ap.add_argument(
        "--per-boundary-box",
        action="store_true",
        help="set C=1/(nu*n_prefix) for each boundary solve",
    )
    ap.add_argument("--solver-seed", type=int, default=0)
    ap.add_argument("--stage1-steps", type=int, default=2000)
    ap.add_argument("--stage2-steps", type=int, default=8000)
    ap.add_argument("--lr1", type=float, default=0.02)
    ap.add_argument("--lr2", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--eval-blocks", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="legacy experiment seed; supplies data/init seeds unless overridden",
    )
    ap.add_argument("--data-seed", type=int)
    ap.add_argument("--init-seed", type=int)
    ap.add_argument("--out", default="outputs/gemma_sv_distill")
    ap.add_argument("--control-only", action="store_true",
                    help="run ONLY the matched control: original (ungrafted) model + identical "
                         "LoRA budget -- isolates the gate's true residual cost vs plain fine-tune")
    ap.add_argument("--compare-to", default="outputs/gemma_sv_distill/results.json",
                    help="grafted run's results.json, to print the gate residual cost")
    ap.add_argument("--data-skip", type=int, default=0,
                    help="advance the data stream N batches before training (to match the "
                         "grafted run's stage-2 batches exactly; 0 = same-distribution seed match)")
    ap.add_argument(
        "--require-matched-stream",
        action="store_true",
        help="fail a control unless --compare-to has an identical stage-2 fingerprint",
    )
    args = ap.parse_args()
    args.data_seed = args.seed if args.data_seed is None else args.data_seed
    args.init_seed = args.seed if args.init_seed is None else args.init_seed
    if args.data_skip < 0:
        ap.error("--data-skip must be non-negative")
    if args.require_matched_stream and not args.control_only:
        ap.error("--require-matched-stream is valid only with --control-only")

    from transformers import AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.data import fineweb_edu_batch_iter, perplexity, wikitext103_blocks
    from gemma_sv.distill import (attention_transfer_loss,
                                  capture_global_reference_outputs, setup_stage1)
    from gemma_sv.layer_select import find_global_attention_layers

    dev, out = args.device, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = seed_everything(args.seed)
    training_steps = (
        args.stage2_steps
        if args.control_only
        else args.stage1_steps + args.stage2_steps
    )
    budget = training_steps * args.batch * args.seq_len
    print(f"=== SV-graft distillation: {args.model} on {dev} | token budget ~{budget/1e6:.1f}M ===")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    ev = wikitext103_blocks(
        tok,
        args.seq_len,
        max_blocks=args.eval_blocks,
        revision=args.wikitext_revision,
    )
    print(f"eval: {len(ev)} WikiText-103 blocks x {args.seq_len} tok")
    eval_sha256 = tensor_sha256(torch.tensor(ev, dtype=torch.long))
    data = fineweb_edu_batch_iter(
        tok,
        args.seq_len,
        args.batch,
        device=dev,
        seed=args.data_seed,
        revision=args.fineweb_revision,
    )
    for _ in range(args.data_skip):                        # align to another run's stage-2 stream
        next(data)

    if args.control_only:                                  # matched baseline: original + LoRA, no graft
        student = _load(args.model, dev, args.model_revision)
        ppl_orig = perplexity(student, ev, device=dev, batch=2)
        print(f"ppl ORIGINAL = {ppl_orig:.2f}")
        seed_everything(args.init_seed)
        student = _lora(student, args.rank, dev).train()
        params = [p for p in student.parameters() if p.requires_grad]
        print(f"control LoRA r={args.rank}: {sum(p.numel() for p in params):,} trainable "
              f"(matched to the grafted stage-2 budget)")
        opt = torch.optim.Adam(params, lr=args.lr2)
        stream = BatchFingerprint()
        res = {
            "schema": "gemma-sv-recovery-v2",
            "mode": "control",
            "model": args.model,
            "ppl_orig": ppl_orig,
            "config": vars(args),
            "rng": rng,
            "eval": {
                "dataset": "Salesforce/wikitext/wikitext-103-raw-v1",
                "revision": args.wikitext_revision,
                "split": "test",
                "blocks": len(ev),
                "seq_len": args.seq_len,
                "token_ids_sha256": eval_sha256,
            },
            "stage2_offset_batches": args.data_skip,
        }

        def s2_loss(ids):
            return student(input_ids=ids, labels=ids).loss

        try:
            for off in range(0, args.stage2_steps, args.ckpt_every):
                n = min(args.ckpt_every, args.stage2_steps - off)
                _train(
                    student,
                    opt,
                    data,
                    n,
                    s2_loss,
                    "control",
                    args.log_every,
                    fingerprint=stream,
                )
                student.save_pretrained(out / "lora_adapter")
                print(f"  ckpt -> {out/'lora_adapter'} ({off + n}/{args.stage2_steps})", flush=True)
        finally:
            ppl_control = perplexity(student, ev, device=dev, batch=2)
            res["ppl_control"] = ppl_control
            res["stage2_stream"] = stream.summary()
            adapter_path = out / "lora_adapter"
            if adapter_path.exists():
                res["adapter_sha256"] = path_sha256(adapter_path)
            pair_error = None
            cmp_path = Path(args.compare_to)
            if cmp_path.exists():
                graft = json.loads(cmp_path.read_text())
                try:
                    validate_stage2_pair(graft, res)
                except ValueError as exc:
                    pair_error = str(exc)
                    res["pairing"] = {
                        "valid": False,
                        "recovery_results": str(cmp_path),
                        "error": pair_error,
                    }
                else:
                    res["pairing"] = {
                        "valid": True,
                        "recovery_results": str(cmp_path),
                    }
            elif args.require_matched_stream:
                pair_error = f"missing recovery result required for pairing: {cmp_path}"
                res["pairing"] = {
                    "valid": False,
                    "recovery_results": str(cmp_path),
                    "error": pair_error,
                }
            write_json_atomic(out / "control_results.json", res)
            print(f"\nppl CONTROL (original + matched LoRA) = {ppl_control:.2f}")
            if cmp_path.exists():
                pf = graft.get("ppl_final")
                if pf:
                    print(f"  grafted+LoRA {pf:.2f}  vs  control {ppl_control:.2f}  =>  gate residual "
                          f"cost {100*(pf/ppl_control-1):+.1f}% ppl (the honest zero-utility-loss number)")
            print(f"  saved -> {out/'control_results.json'}")
            if pair_error and args.require_matched_stream:
                raise ValueError(pair_error)
        return 0

    student = _load(args.model, dev, args.model_revision)
    ppl_orig = perplexity(student, ev, device=dev, batch=2)
    print(f"ppl ORIGINAL = {ppl_orig:.2f}")

    teacher = None
    if args.stage1_steps > 0:
        teacher = _load(args.model, dev, args.model_revision).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    layer_ids = [i for i, _ in find_global_attention_layers(student)]
    graft_sv_into_gemma(
        student,
        nu=args.nu,
        chunk=args.chunk,
        readout="softmax",
        preserve_prefix_mass=args.preserve_prefix_mass,
        per_boundary_box=args.per_boundary_box,
        solver_seed=args.solver_seed,
    )
    ppl_graft0 = perplexity(student, ev, device=dev, batch=2)
    print(f"ppl GRAFTED (untrained) = {ppl_graft0:.2f}  (gate cost +{100*(ppl_graft0/ppl_orig-1):.1f}%)")

    stage2_offset = args.data_skip + args.stage1_steps + (
        1 if args.stage1_steps > 0 else 0
    )
    stream = BatchFingerprint()
    results = {
        "schema": "gemma-sv-recovery-v2",
        "mode": "recovered",
        "model": args.model,
        "ppl_orig": ppl_orig,
        "ppl_graft0": ppl_graft0,
        "config": vars(args),
        "rng": rng,
        "eval": {
            "dataset": "Salesforce/wikitext/wikitext-103-raw-v1",
            "revision": args.wikitext_revision,
            "split": "test",
            "blocks": len(ev),
            "seq_len": args.seq_len,
            "token_ids_sha256": eval_sha256,
        },
        "stage2_offset_batches": stage2_offset,
    }
    recovery_state = None
    try:
        if args.stage1_steps > 0:                          # stage 1: attention transfer (kpar)
            kpar_params = setup_stage1(student, {"input_ids": next(data)})
            student.to(dev)
            opt1 = torch.optim.Adam(kpar_params, lr=args.lr1)

            def s1_loss(ids):
                refs = capture_global_reference_outputs(teacher, {"input_ids": ids}, layer_ids)
                return attention_transfer_loss(student, {"input_ids": ids}, refs)

            _train(student, opt1, data, args.stage1_steps, s1_loss, "stage1", args.log_every)
            del teacher
            results["kpar"] = {i: dict(find_global_attention_layers(student))[i]
                               .self_attn.log_kpar.exp().item() for i in layer_ids}
            recovery_state = collect_recovery_state(
                student,
                model_id=args.model,
                model_revision=args.model_revision,
            )

        seed_everything(args.init_seed)
        student = _lora(student, args.rank, dev).train()   # stage 2: LoRA recovery (LM loss)
        lora_params = [p for p in student.parameters() if p.requires_grad]
        n_tr = sum(p.numel() for p in lora_params)
        print(f"stage2 LoRA r={args.rank}: {n_tr:,} trainable")
        opt2 = torch.optim.Adam(lora_params, lr=args.lr2)

        def s2_loss(ids):
            return student(input_ids=ids, labels=ids).loss

        t0 = time.time()
        for off in range(0, args.stage2_steps, args.ckpt_every):
            n = min(args.ckpt_every, args.stage2_steps - off)
            _train(
                student,
                opt2,
                data,
                n,
                s2_loss,
                "stage2",
                args.log_every,
                fingerprint=stream,
            )
            student.save_pretrained(out / "lora_adapter")
            if recovery_state is not None:
                save_recovery_state(out / "lora_adapter", recovery_state)
            print(f"  ckpt -> {out/'lora_adapter'} ({off + n}/{args.stage2_steps})", flush=True)
        results["stage2_minutes"] = (time.time() - t0) / 60
    finally:
        ppl_final = perplexity(student, ev, device=dev, batch=2)
        results["ppl_final"] = ppl_final
        results["stage2_stream"] = stream.summary()
        if recovery_state is not None:
            results["recovery_state"] = recovery_state
        adapter_path = out / "lora_adapter"
        if adapter_path.exists():
            results["adapter_sha256"] = path_sha256(adapter_path)
        write_json_atomic(out / "results.json", results)
        print(f"\nppl FINAL (grafted+recovered) = {ppl_final:.2f}")
        print(f"  original {ppl_orig:.2f} -> graft {ppl_graft0:.2f} -> recovered {ppl_final:.2f}  "
              f"(parity gap {100*(ppl_final/ppl_orig-1):+.1f}% vs original)")
        print(f"  saved -> {out/'results.json'}")
        print("  NOTE: rigorous parity also needs the --control (original+LoRA) to separate "
              "gate-recovery from plain fine-tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

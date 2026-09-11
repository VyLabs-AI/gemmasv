"""Exact in-context unlearning at the MODEL's NEXT-TOKEN OUTPUT on real gemma-3-1b.

`unlearn_demo` certifies exactness at the gate READOUT (per layer). This lifts it to the
model's actual next-token logits -- the behavioral claim TOFU/MUSE-style evals demand
("the model forgot" is a statement about model output, not a hidden activation):

  forgetting a long-range-memory token via the float64 C&P DECREMENT leaves the model's
  next-token distribution close to its fixed-C RETAINED-KEY REFIT,
  while DECAY (keep it, alpha*=0.01) and KEEP both visibly change the output.

Faithful: the exact gate alphas are computed by the float64 C&P solver (cp_svm) -- decrement,
refit-without, decay -- and threaded into the live readout via ``alpha_override`` (FISTA is
skipped), then the FULL model runs in float64 -> logits. One gated boundary (chunk = T/2) keeps
the gate a single clean prefix gate per layer.

``--trials N`` reports the KL DISTRIBUTION over N forget targets (median / worst case), not a
single deletion -- mirroring the capability paper's many-trial rigor.

Run: .venv311/bin/python -m gemma_sv.unlearn_output_demo
     .venv311/bin/python -m gemma_sv.unlearn_output_demo --lora outputs/gemma_sv_distill/lora_adapter --trials 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import warnings

import numpy as np
import torch

from gemma_sv.recovery_protocol import MODEL_REVISION
from gemma_sv.recovery_state import apply_recovery_state

warnings.filterwarnings("ignore")

MODEL_ID = "google/gemma-3-1b-pt"
NU = 0.3
T = 192            # sequence length
CHUNK = 96         # one gated boundary at CHUNK -> queries [CHUNK:T] gate over prefix [0:CHUNK]
DECAY = 0.01       # approximate-unlearning foil
PROMPT = "In a faraway kingdom, a curious fox kept a careful ledger of every promise. " * 16
# Non-repetitive control (--diverse): rules out repeated-prompt ill-conditioning as the source of
# elevated KL (the 12B scale finding was confirmed on both prompts).
PROMPT_DIVERSE = (
    "The harbor town woke to gulls and diesel. Mira counted the crates twice, then "
    "signed the manifest in green ink. Down the quay, an old crane groaned against a "
    "load of citrus bound for the northern markets, where winter had already bitten "
    "the orchards. A child chased a paper boat along the gutter. The tide turned at "
    "noon; by three the fog had swallowed the lighthouse, and the radio gave warnings "
    "in three languages. Somewhere a violin practiced the same difficult bar, again "
    "and again, never quite landing the final note before the silence took it back. "
) * 3


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_path(raw_path: str | None) -> str | None:
    """Hash a local adapter file/tree without depending on directory metadata."""
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        return None

    digest = hashlib.sha256(b"gemma-sv-path-v1\0")
    files = (
        [path]
        if path.is_file()
        else sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    )
    for candidate in files:
        relative = (
            candidate.name if path.is_file() else candidate.relative_to(path).as_posix()
        ).encode("utf-8")
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _provenance(model_id: str, adapter: str | None, model) -> dict:
    config = model.config.to_dict()
    revision = getattr(model.config, "_commit_hash", None)
    model_descriptor = {
        "id": model_id,
        "revision": revision,
        "content_sha256": _hash_path(model_id),
        "config_sha256": _sha256_bytes(_canonical_bytes(config)),
    }
    adapter_configs = {}
    for name, adapter_config in sorted(getattr(model, "peft_config", {}).items()):
        to_dict = getattr(adapter_config, "to_dict", None)
        adapter_configs[str(name)] = to_dict() if to_dict else str(adapter_config)
    adapter_descriptor = {
        "path": adapter,
        "content_sha256": _hash_path(adapter),
        "config_sha256": (
            _sha256_bytes(_canonical_bytes(adapter_configs)) if adapter_configs else None
        ),
    }
    model_descriptor["descriptor_sha256"] = _sha256_bytes(
        _canonical_bytes(model_descriptor)
    )
    adapter_descriptor["descriptor_sha256"] = (
        _sha256_bytes(_canonical_bytes(adapter_descriptor)) if adapter else None
    )
    model_descriptor["sha256"] = (
        model_descriptor["content_sha256"]
        or model_descriptor["descriptor_sha256"]
    )
    adapter_descriptor["sha256"] = (
        adapter_descriptor["content_sha256"]
        or adapter_descriptor["descriptor_sha256"]
    )
    return {
        "model": model_descriptor,
        "adapter": adapter_descriptor,
        "script_sha256": _sha256_bytes(Path(__file__).read_bytes()),
    }


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _select_targets(
    support: list[int],
    trials: int,
    target_seed: int,
) -> list[int]:
    """Reproduce the old seed-0 choices while allowing deterministic alternatives."""
    if not support:
        return [5]
    if trials <= 1:
        if target_seed == 0:
            return [support[len(support) // 2]]
        rng = np.random.default_rng(target_seed)
        return [int(rng.choice(support))]
    rng = np.random.default_rng(target_seed)
    return [
        int(value)
        for value in rng.choice(
            support,
            size=min(trials, len(support)),
            replace=False,
        )
    ]


def _kpar(X: np.ndarray) -> float:
    d2 = np.sum((X[:, None] - X[None]) ** 2, axis=-1)
    return float(np.sqrt(np.median(d2[d2 > 0])))


def _cp_alphas(Xh: np.ndarray, i: int, kept: list, C: float):
    """float64 C&P alphas for one head's prefix keys: (full, decrement, refit-without)."""
    from cp_svm import FastOneClassSVM

    kp = _kpar(Xh)
    full = np.asarray(FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(Xh).alpha)
    m = FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(Xh)
    m.remove_point(i)                                          # EXACT decremental unlearning
    a_dec = np.asarray(m.alpha)
    a_ref = np.asarray(FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(Xh[kept]).alpha)
    return full, a_dec, a_ref


def _overrides(caps, layer_ids, i: int, C: float, H: int, b: int):
    """Per-case ({keep,forget,refit,decay}) per-layer (H, b) gate-alpha overrides for token i."""
    kept = [j for j in range(b) if j != i]
    cases = {k: {} for k in ("keep", "forget", "refit", "decay")}
    for idx in layer_ids:
        K = caps[idx]
        keep, forget, refit, decay = (np.zeros((H, b)) for _ in range(4))
        for h in range(H):
            full, a_dec, a_ref = _cp_alphas(K[h], i, kept, C)
            keep[h] = full
            forget[h, kept] = a_dec                            # token i stays 0 (forgotten)
            refit[h, kept] = a_ref                             # token i never in the gate
            decay[h] = full.copy()
            decay[h, i] *= DECAY
        for nm, arr in (("keep", keep), ("forget", forget), ("refit", refit), ("decay", decay)):
            cases[nm][idx] = arr
    return cases


def _overrides_set(caps, layer_ids, F, C: float, H: int, b: int):
    """Overrides for forgetting a SET F: sequential C&P decrements (exact) vs refit-without-F
    (never-had-them) vs decay (keep all of F, alpha*=DECAY). Used for the sustainability sweep."""
    from cp_svm import FastOneClassSVM

    Fs = set(F)
    kept = [j for j in range(b) if j not in Fs]
    cases = {k: {} for k in ("exact", "refit", "decay")}
    for idx in layer_ids:
        K = caps[idx]
        exact, refit, decay = (np.zeros((H, b)) for _ in range(3))
        for h in range(H):
            kp = _kpar(K[h])
            full = np.asarray(FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(K[h]).alpha)
            m = FastOneClassSVM(C=C, ktype="r", kpar=kp).seed_from_qp(K[h])
            for j in sorted(F, reverse=True):                  # sequential exact decrements
                m.remove_point(j)
            exact[h, kept] = np.asarray(m.alpha)
            refit[h, kept] = np.asarray(FastOneClassSVM(C=C, ktype="r", kpar=kp)
                                        .seed_from_qp(K[h][kept]).alpha)
            decay[h] = full.copy()
            decay[h, list(F)] *= DECAY
        for nm, arr in (("exact", exact), ("refit", refit), ("decay", decay)):
            cases[nm][idx] = arr
    return cases


def main(argv=None) -> int:
    from cp_svm import FastOneClassSVM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument(
        "--model-revision",
        default=None,
        help="Hugging Face revision (the default 1B model is pinned automatically)",
    )
    ap.add_argument("--lora", default=None, help="stage-2 adapter -> run on the RECOVERED model")
    ap.add_argument("--trials", type=int, default=1, help="N forget targets -> KL distribution")
    ap.add_argument(
        "--target-seed",
        type=int,
        default=0,
        help="deterministic seed for forget-target selection (0 preserves legacy choices)",
    )
    ap.add_argument(
        "--run-seed",
        "--training-seed",
        dest="run_seed",
        type=int,
        default=None,
        help="training-seed label recorded in JSON for multiseed pairing",
    )
    ap.add_argument("--sequential", default=None,
                    help="comma k-values (e.g. 1,2,5,10,20): measured numerical stability sweep")
    ap.add_argument("--diverse", action="store_true",
                    help="use the non-repetitive control prompt (prompt-artifact check)")
    ap.add_argument(
        "--json-out",
        "--json",
        dest="json_out",
        default=None,
        help="optional path for a deterministic machine-readable result",
    )
    args = ap.parse_args(argv)
    if args.model_revision is None and args.model == MODEL_ID:
        args.model_revision = MODEL_REVISION

    tag = "recovered: graft+LoRA" if args.lora else "untrained graft"
    print(f"=== exact unlearning @ NEXT-TOKEN OUTPUT: {args.model} [{tag}], float64/CPU ===")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=torch.float64,
    ).eval()
    graft_sv_into_gemma(model, nu=NU, chunk=CHUNK, readout="softmax")
    recovery_state = None
    if args.lora:
        from peft import PeftModel

        recovery_state = apply_recovery_state(model, args.lora, strict=False)
        # PEFT auto-dispatches the adapter to MPS, which has no float64; hide MPS for the load.
        _mps = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        try:
            model = PeftModel.from_pretrained(model, args.lora).to(torch.float64).eval()
        finally:
            torch.backends.mps.is_available = _mps

    glayers = dict(find_global_attention_layers(model))
    layer_ids = sorted(glayers)
    ids = tok(PROMPT_DIVERSE if args.diverse else PROMPT, return_tensors="pt").input_ids[:, :T]
    assert ids.shape[1] == T, f"prompt too short: {ids.shape[1]} < {T}"

    caps = {}
    handles = []
    for idx in layer_ids:
        attn = glayers[idx].self_attn

        def _hook(mod, a, kw, _idx=idx, _attn=attn):
            hs = a[0] if a else kw["hidden_states"]
            _, k, _ = _attn._project_qkv(hs, kw.get("position_embeddings"))
            caps[_idx] = k[0, :, :CHUNK].detach().cpu().numpy()    # (H, CHUNK, d)

        handles.append(attn.register_forward_pre_hook(_hook, with_kwargs=True))
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    H, b = caps[layer_ids[0]].shape[0], CHUNK
    C = 1.0 / (NU * T)                                          # the box the gate uses (n := T)
    cand = set()                                               # forget targets = union of support
    for idx in layer_ids:                                      # tokens across all layers/heads
        for h in range(H):
            X = caps[idx][h]
            cand.update(int(s) for s in
                        FastOneClassSVM(C=C, ktype="r", kpar=_kpar(X)).seed_from_qp(X).S)
    S = sorted(cand)
    report = {
        "schema_version": 1,
        "evaluation": "gemma_sv_output_unlearning",
        "seed": args.run_seed,
        "provenance": _provenance(args.model, args.lora, model),
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter": args.lora,
            "trials": args.trials,
            "target_seed": args.target_seed,
            "run_seed": args.run_seed,
            "sequential": args.sequential,
            "prompt": "diverse" if args.diverse else "repeated",
            "prompt_sha256": _sha256_bytes(
                (PROMPT_DIVERSE if args.diverse else PROMPT).encode("utf-8")
            ),
            "nu": NU,
            "sequence_length": T,
            "chunk": CHUNK,
            "decay": DECAY,
            "dtype": "float64",
            "device": "cpu",
            "recovery_state_restored": recovery_state is not None,
        },
        "support": {
            "count": len(S),
            "indices_sha256": _sha256_bytes(_canonical_bytes(S)),
            "global_layer_ids": layer_ids,
            "heads": H,
            "prefix_tokens": b,
        },
    }

    def run_case(ov):
        for idx in layer_ids:
            glayers[idx].self_attn._alpha_override = {CHUNK: ov[idx]}
        with torch.no_grad():
            lg = model(ids).logits[0, CHUNK:T].double()
        for idx in layer_ids:
            glayers[idx].self_attn._alpha_override = None
        return lg

    def md(a, ref):
        return float((a - ref).abs().max())

    def kl(a, ref):
        la, lr = torch.log_softmax(a, -1), torch.log_softmax(ref, -1)
        return max(0.0, float((la.exp() * (la - lr)).sum(-1).mean()))

    if args.sequential:                                         # sustainability: KL vs #deletions
        ks = [int(x) for x in args.sequential.split(",") if int(x) <= len(S)]
        F_all = (
            S[:max(ks)]
            if args.target_seed == 0
            else _select_targets(S, max(ks), args.target_seed)
        )
        print(f"sequential deletions over {len(S)} support tokens; k in {ks}\n")
        print(f"{'k deletes':>10}{'KL(exact||refit)':>20}{'KL(decay||refit)':>20}")
        kle, kld, rows = [], [], []
        for k in ks:
            cs = _overrides_set(caps, layer_ids, F_all[:k], C, H, b)
            Lr = run_case(cs["refit"])
            e, d = kl(run_case(cs["exact"]), Lr), kl(run_case(cs["decay"]), Lr)
            kle.append(e)
            kld.append(d)
            rows.append(
                {
                    "deletion_count": k,
                    "target_indices": F_all[:k],
                    "kl_exact_vs_refit_nats": e,
                    "kl_decay_vs_refit_nats": d,
                    "decrement_fallbacks": 0,
                }
            )
            print(f"{k:>10}{e:>20.2e}{d:>20.2e}", flush=True)
        print(f"\ndecrement/refit KL reaches {max(kle):.1e}; "
              f"decay reaches {kld[-1]:.1e} ({kld[-1]/max(kle[-1],1e-300):.0e}x at k={ks[-1]})")
        assert np.isfinite(kle).all() and np.isfinite(kld).all()
        assert np.median(kle) < np.median(kld), "median decrement/refit KL does not beat decay"
        print("SEQUENTIAL DIAGNOSTIC — decrement/refit remains below decay on the median; "
              "inspect the reported numerical floor rather than assuming a fixed threshold.")
        report.update(
            {
                "mode": "sequential",
                "sequential_rows": rows,
                "summary": {
                    "n_steps": len(rows),
                    "worst_kl_exact_vs_refit_nats": float(max(kle)),
                    "worst_kl_decay_vs_refit_nats": float(max(kld)),
                    "decrement_fallbacks": 0,
                },
            }
        )
        _write_json(args.json_out, report)
        return 0

    if args.trials <= 1:                                        # single deletion, detailed table
        i = _select_targets(S, 1, args.target_seed)[0]
        L = {k: run_case(v) for k, v in _overrides(caps, layer_ids, i, C, H, b).items()}
        print(f"forget token i={i} (support), {H} heads, prefix={b}, gated block [{CHUNK}:{T}]\n")
        print(f"{'compare':<20}{'max|Δlogit|':>14}{'mean KL (nats)':>16}")
        for nm in ("forget", "decay", "keep"):
            print(f"{nm + ' vs refit':<20}{md(L[nm], L['refit']):>14.2e}{kl(L[nm], L['refit']):>16.2e}")
        kl_fr, kl_dr = kl(L["forget"], L["refit"]), kl(L["decay"], L["refit"])
        d_fr, d_dr, d_kr = (md(L[x], L["refit"]) for x in ("forget", "decay", "keep"))
        print(f"\ndecrement vs fixed-C retained-key refit at the next-token distribution: "
              f"KL={kl_fr:.2e} nats, "
              f"vs decay KL={kl_dr:.2e} ({kl_dr/max(kl_fr,1e-300):.0e}x larger)")
        assert np.isfinite([kl_fr, kl_dr, d_fr, d_dr, d_kr]).all()
        print("OUTPUT-LEVEL DIAGNOSTIC — report the measured decrement/refit KL and contrast; "
              "the realized numerical floor is model- and target-dependent.")
        report.update(
            {
                "mode": "single_target",
                "targets": [
                    {
                        "target_index": i,
                        "kl_exact_vs_refit_nats": kl_fr,
                        "kl_decay_vs_refit_nats": kl_dr,
                        "max_abs_logit_exact_vs_refit": d_fr,
                        "max_abs_logit_decay_vs_refit": d_dr,
                        "max_abs_logit_keep_vs_refit": d_kr,
                        "decrement_fallbacks": 0,
                    }
                ],
                "summary": {
                    "n_targets": 1,
                    "mean_kl_exact_vs_refit_nats": kl_fr,
                    "median_kl_exact_vs_refit_nats": kl_fr,
                    "worst_kl_exact_vs_refit_nats": kl_fr,
                    "mean_kl_decay_vs_refit_nats": kl_dr,
                    "median_kl_decay_vs_refit_nats": kl_dr,
                    "minimum_kl_decay_vs_refit_nats": kl_dr,
                    "decrement_fallbacks": 0,
                },
            }
        )
        _write_json(args.json_out, report)
        return 0

    # Distribution over many forget targets.
    targets = _select_targets(S, args.trials, args.target_seed)
    klf, kld, rows = [], [], []
    for n, i in enumerate(targets):
        cs = _overrides(caps, layer_ids, i, C, H, b)
        Lr = run_case(cs["refit"])
        exact_logits = run_case(cs["forget"])
        decay_logits = run_case(cs["decay"])
        exact_kl = kl(exact_logits, Lr)
        decay_kl = kl(decay_logits, Lr)
        klf.append(exact_kl)
        kld.append(decay_kl)
        rows.append(
            {
                "target_index": i,
                "kl_exact_vs_refit_nats": exact_kl,
                "kl_decay_vs_refit_nats": decay_kl,
                "max_abs_logit_exact_vs_refit": md(exact_logits, Lr),
                "max_abs_logit_decay_vs_refit": md(decay_logits, Lr),
                "decrement_fallbacks": 0,
            }
        )
        if n % 10 == 0 or n == len(targets) - 1:
            print(f"  trial {n + 1}/{len(targets)} (forget tok {i})  KL_forget={klf[-1]:.1e}  "
                  f"KL_decay={kld[-1]:.1e}", flush=True)
    klf, kld = np.array(klf), np.array(kld)
    print(f"\n{len(targets)} forget targets (support tokens), next-token output KL vs refit-without:")
    print(f"  KL(forget||refit): median={np.median(klf):.1e}  mean={klf.mean():.1e}  WORST={klf.max():.1e}")
    print(f"  KL(decay ||refit): median={np.median(kld):.1e}  min={kld.min():.1e}")
    print(f"  worst exact KL {klf.max():.1e}  <<  min decay leak {kld.min():.1e}  "
          f"({kld.min()/max(klf.max(),1e-300):.0e}x)")
    assert np.isfinite(klf).all() and np.isfinite(kld).all()
    assert np.median(klf) < np.median(kld), "median decrement/refit KL does not beat decay"
    if klf.max() < kld.min() / 100:
        print("OUTPUT-LEVEL DISTRIBUTION — every measured decrement is strongly separated from decay.")
    else:
        print("OUTPUT-LEVEL DISTRIBUTION — median decrement/refit KL beats decay, but distribution "
              "edges overlap; report the measured numerical floor.")
    report.update(
        {
            "mode": "target_distribution",
            "targets": rows,
            "summary": {
                "n_targets": len(rows),
                "mean_kl_exact_vs_refit_nats": float(klf.mean()),
                "median_kl_exact_vs_refit_nats": float(np.median(klf)),
                "worst_kl_exact_vs_refit_nats": float(klf.max()),
                "mean_kl_decay_vs_refit_nats": float(kld.mean()),
                "median_kl_decay_vs_refit_nats": float(np.median(kld)),
                "minimum_kl_decay_vs_refit_nats": float(kld.min()),
                "worst_kl_decay_vs_refit_nats": float(kld.max()),
                "decrement_fallbacks": 0,
            },
        }
    )
    _write_json(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

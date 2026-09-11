"""Smoke test for the SV graft into Gemma 3 (see docs/distillation_followup_plan.md §5).

Runs the four-step first-run check WITHOUT the gated Gemma download: a tiny,
random-init ``Gemma3ForCausalLM`` carrying the real 5:1 interleave exercises the
exact code paths we care about for "is the scaffold wired correctly" --

    1. summarize_layer_types  : confirm the ``L L L L L G`` pattern + global count
    2. graft_sv_into_gemma    : swap the global layers' self_attn for the SV gate
    3. one tiny forward       : RoPE-on-keys, GQA expand, base scaling, the batched
                                FISTA gate solve -- shapes finite, no exceptions
    4. exact_forget_equivalence: float64 C&P decrement vs fixed-C refit on a
                                grafted global layer's live keys -- the active-set
                                partition (S,E,R) must match EXACTLY, and the
                                decision-function deviation must be at machine-
                                precision-ish (the QP-seed/RBF-conditioning floor;
                                ~1e-7 on these random-init keys, ~1e-13 when the
                                gate Gram is well-conditioned -- cf. forgetting_rigor)
    5. persistent prefill     : ingest once, snapshot native/SV caches, and query
                                two independent forks without refeeding the prefix

Green == all five asserts pass: the graft is wired correctly, the forward is finite,
and decrement is refit-equivalent to logged tolerance. Real pretrained
weights (gated, license + HF token) are needed only for quality/training -- NOT for
this plumbing check.

Run:  .venv311/bin/python -m gemma_sv.smoke_test
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Tiny config knobs -- big enough to fire the gate (prefix >= ceil(1/C)+2 past one
# chunk boundary), small enough to run in a second on CPU.
N_LAYERS = 12          # 5:1 interleave -> 2 global layers (pattern repeats once)
HIDDEN = 64
N_HEADS = 4
N_KV_HEADS = 1         # GQA: 4 query heads share 1 KV head (exercises repeat_kv)
HEAD_DIM = 16
SEQ_LEN = 32
CHUNK = 8              # small so a gated boundary lands inside SEQ_LEN
NU = 0.3
SEED = 0
# "Machine-precision-ish" bar for the decision-function deviation. The exact claim
# is two-part: (a) the active-set partition matches the from-scratch refit EXACTLY
# (deterministic), and (b) f_dev sits at the QP-seed/RBF-conditioning floor -- ~1e-7
# on these random-init keys, vs the decay baseline's O(1) leak. forgetting_rigor sees
# median ~1e-9 with an ill-conditioned tail; 1e-5 gives deterministic headroom while
# staying >=4 orders below decay.
EXACT_TOL = 1e-5


def build_tiny_gemma():
    """A random-init Gemma3 text model with the real global/local interleave."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=256, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS, num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        sliding_window=16, max_position_embeddings=128,
    )
    torch.manual_seed(SEED)
    return Gemma3ForCausalLM(cfg).eval(), cfg


def step1_layer_types(model) -> list[int]:
    from gemma_sv import summarize_layer_types
    from gemma_sv.layer_select import find_global_attention_layers

    summary = summarize_layer_types(model)
    globals_ = [i for i, _ in find_global_attention_layers(model)]
    print(f"[1] {summary}")
    print(f"    global layer indices: {globals_}")
    seq = summary.split(": ", 1)[1]
    assert "G" in seq, "no global layers detected"
    assert seq.startswith("L L L L L G"), f"unexpected interleave: {seq!r}"
    assert globals_ == list(range(5, N_LAYERS, 6)), f"global idx mismatch: {globals_}"
    return globals_


def step2_graft(model, globals_) -> None:
    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.layer_select import find_global_attention_layers
    from gemma_sv.sv_global_attention import SVGlobalAttention

    replaced = graft_sv_into_gemma(model, nu=NU, chunk=CHUNK, readout="softmax")
    print(f"[2] grafted SV gate into global layers {replaced}")
    assert replaced == globals_, f"grafted {replaced} != globals {globals_}"
    # The grafted layers must STILL be discoverable as global layers (the distill
    # hooks + key extraction depend on this); guards against a vacuous check.
    post = find_global_attention_layers(model)
    assert [i for i, _ in post] == globals_, \
        f"grafted globals not rediscoverable: {[i for i, _ in post]} != {globals_}"
    for idx, layer in post:
        assert isinstance(layer.self_attn, SVGlobalAttention), \
            f"layer {idx} self_attn was not replaced"
    print(f"    base scaling on a grafted layer: "
          f"{dict(post)[globals_[0]].self_attn.base.scaling:.4g} "
          f"(== query_pre_attn_scalar**-0.5)")


def step3_forward(model) -> torch.Tensor:
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 256, (1, SEQ_LEN))
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits
    print(f"[3] forward ok: logits {tuple(logits.shape)}, "
          f"finite={bool(torch.isfinite(logits).all())}")
    assert logits.shape == (1, SEQ_LEN, 256), f"bad logits shape {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "non-finite logits from grafted forward"
    return input_ids


def step4_exact_forget(model, input_ids, layer_idx: int) -> float:
    from gemma_sv import exact_forget_equivalence
    from gemma_sv.unlearn import certified_reserve, extract_global_layer_keys

    keys = extract_global_layer_keys(model, input_ids, layer_idx, batch=0, head=0)
    # RBF bandwidth from the live keys (median pairwise distance) -> well-posed gate.
    d2 = np.sum((keys[:, None] - keys[None]) ** 2, axis=-1)
    kpar = float(np.sqrt(np.median(d2[d2 > 0])))
    forget = [5, 20]
    res = exact_forget_equivalence(keys, forget, nu=NU, kpar=kpar)
    reserve = certified_reserve(keys, nu=NU, kpar=kpar)
    print(f"[4] global layer {layer_idx}: keys {keys.shape}, kpar={kpar:.3g}, "
          f"forget={forget}")
    print(f"    decrement/refit f_dev={res['f_dev']:.2e}  "
          f"partition_match={res['partition_match']:.0f}  "
          f"C={res['C']:.4g}  reserve(|R|)={len(reserve)}")
    assert res["partition_match"] == 1.0, \
        "decrement active-set (S,E,R) != fixed-C refit"
    assert res["f_dev"] < EXACT_TOL, \
        f"decrement/refit deviation={res['f_dev']:.2e} >= {EXACT_TOL:.0e}"
    return res["f_dev"]


def step5_persistent_prefill(model, input_ids) -> None:
    from types import SimpleNamespace

    from gemma_sv.demo_server.gate_context import ModelGateController
    from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
    from gemma_sv.layer_select import find_global_attention_layers

    class TinyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            ids = [1 + (ord(char) % 250) for char in text]
            return SimpleNamespace(input_ids=ids)

        @staticmethod
        def decode(ids):
            return " ".join(str(int(token)) for token in ids)

    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id="random-gemma-smoke",
            lora_path=None,
            device="cpu",
            dtype="float32",
            generation_tokens=2,
            window=16,
        )
    )
    runtime.model = model
    runtime.tokenizer = TinyTokenizer()
    runtime.layers = dict(find_global_attention_layers(model))
    runtime.controller = ModelGateController(runtime.layers)
    runtime.loaded = True
    memory_ids = [int(value) for value in input_ids[0].tolist()]
    persistent = runtime.prefill_persistent(memory_ids)
    first_text, first = runtime.generate_persistent(persistent, "query")
    second_text, second = runtime.generate_persistent(persistent, "query")
    assert first_text == second_text
    assert np.array_equal(first, second)
    assert persistent.token_count == len(memory_ids)
    assert all(session.frozen for session in persistent.layer_sessions.values())
    overrides, states = runtime.persistent_certificate_states(
        persistent, (5,)
    )
    exact = runtime.score_persistent(states["exact"], "query", [1])
    refit = runtime.score_persistent(states["refit"], "query", [1])
    assert np.isfinite(exact["mean_log_probability"])
    assert np.isfinite(refit["mean_log_probability"])
    assert overrides["box_C"] == next(
        iter(persistent.layer_sessions.values())
    ).box_C
    print(
        "[5] persistent prefill ok: one prefix digest "
        f"{persistent.input_digest[:12]}, deterministic forked queries and "
        "query-independent certificate states"
    )


def main() -> int:
    print("=== gemma_sv smoke test (random-init Gemma3, no gated download) ===")
    model, _ = build_tiny_gemma()
    globals_ = step1_layer_types(model)
    step2_graft(model, globals_)
    input_ids = step3_forward(model)
    f_dev = step4_exact_forget(model, input_ids, globals_[0])
    step5_persistent_prefill(model, input_ids)
    print(f"\nSMOKE TEST GREEN — interleave + graft + finite forward + fixed-C "
          f"refit agreement + persistent prefill (f_dev={f_dev:.1e}, at the "
          f"QP-seed/conditioning floor).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

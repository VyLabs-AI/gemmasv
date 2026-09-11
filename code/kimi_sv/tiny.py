"""A miniature randomly-initialized Kimi Linear model for wiring tests.

This exercises the real ``mlx_lm`` Kimi Linear code path -- KDA layers, global
MLA layers, MoE routing, and the hybrid cache -- at a size that runs in
milliseconds and needs no weight download. It verifies mechanics, never language
quality, mirroring the random-model tier of the Gemma harness.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import mlx.core as mx
from mlx_lm.models.kimi_linear import Model, ModelArgs

# Layer 4 is the lone global MLA layer, so the 3:1 KDA/MLA interleave of the
# released model is reproduced at the smallest size that still has both kinds.
KDA_LAYERS = [1, 2, 3]
FULL_ATTN_LAYERS = [4]


def tiny_args(**overrides: Any) -> ModelArgs:
    cfg = dict(
        model_type="kimi_linear",
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=256,
        head_dim=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": 2,
            # The KDA Metal kernel slices the key dim 32 ways, so head_dim must
            # stay a multiple of 32.
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "kda_layers": KDA_LAYERS,
            "full_attn_layers": FULL_ATTN_LAYERS,
        },
        model_max_length=4096,
        num_experts=4,
        moe_intermediate_size=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        mla_use_nope=True,
        num_experts_per_token=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return ModelArgs(**cfg)


def build_tiny(seed: int = 0, **overrides: Any) -> Tuple[Model, ModelArgs]:
    """Construct a tiny Kimi Linear model with reproducible random weights."""
    mx.random.seed(seed)
    args = tiny_args(**overrides)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model, args


def token_block(n: int, start: int = 0, vocab: int = 512) -> mx.array:
    """A deterministic token block, shaped ``(1, n)``."""
    return mx.array([[(start + i * 7 + 3) % vocab for i in range(n)]])


def ingest(model: Model, cache: List[Any], tokens: mx.array) -> mx.array:
    """Run ``tokens`` through ``model``, updating ``cache``; return final logits."""
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    return logits[:, -1, :]


def greedy(model: Model, cache: List[Any], prompt: mx.array, n: int) -> List[int]:
    """Greedy-decode ``n`` tokens, advancing ``cache``."""
    out: List[int] = []
    logits = ingest(model, cache, prompt)
    for _ in range(n):
        tok = int(mx.argmax(logits, axis=-1).item())
        out.append(tok)
        logits = ingest(model, cache, mx.array([[tok]]))
    return out

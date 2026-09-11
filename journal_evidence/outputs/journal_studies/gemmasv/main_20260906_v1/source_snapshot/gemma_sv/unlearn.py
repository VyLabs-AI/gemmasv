"""Fixed-C deletion audits on a grafted Gemma SV global layer.

The ordinary forward uses a single-precision FISTA solve for speed. The
certificate is a separate computation: the Cauwenberghs--Poggio decremental
reverse path runs in float64 (``cp_svm.FastOneClassSVM.remove_point``) and is
compared with refitting the retained keys. FISTA is never used to certify
deletion. Reserve removal is separately certified only for the currently solved
context (``alpha = 0``); future admissions are outside that statement.

``extract_global_layer_keys`` captures the live model keys;
``exact_forget_equivalence`` and ``certified_reserve`` are NumPy/``cp_svm``
checks used by validation and evaluation scripts.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def exact_forget_equivalence(keys: np.ndarray, forget: List[int], *,
                             nu: float = 0.3, kpar: float = 2.0) -> Dict[str, float]:
    """Check decrement/refit equivalence on one (sequence, head) key set.

    Mirrors ``experiments/forgetting_rigor.one_trial``: decrement the forgotten keys
    via the algebraic C&P reverse path, refit from scratch on the retained keys with the
    SAME box ``C = 1/(nu*n)``, and compare DECISION FUNCTIONS on probes (retained +
    forgotten + jittered). All float64. Returns the function-level deviation -- the
    measured numerical deviation and the state-partition match.
    """
    from cp_svm import FastOneClassSVM

    X = np.atleast_2d(np.asarray(keys, dtype=np.float64))
    n = len(X)
    C = 1.0 / (nu * n)                                 # same box for both solves
    keep = [i for i in range(n) if i not in set(forget)]

    m = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X)
    for c in sorted(forget, reverse=True):
        m.remove_point(c)                              # algebraic fixed-C decrement
    f = FastOneClassSVM(C=C, ktype="r", kpar=kpar).seed_from_qp(X[keep])  # retrain-without

    rng = np.random.default_rng(0)
    Xk = X[keep]
    probes = np.vstack([Xk, X[forget], Xk + 0.1 * rng.standard_normal(Xk.shape)])
    f_dev = float(np.max(np.abs(m.decision_function(probes) - f.decision_function(probes))))
    partition_match = float((set(m.S) == set(f.S)) and (set(m.E) == set(f.E)))
    return {"f_dev": f_dev, "partition_match": partition_match, "C": float(C), "n": float(n)}


def certified_reserve(keys: np.ndarray, *, nu: float = 0.3, kpar: float = 2.0) -> List[int]:
    """Current reserve indices (alpha=0); removing them preserves this readout.

    The result is point-in-time and does not quantify over later admissions.
    """
    from cp_svm import FastOneClassSVM

    X = np.atleast_2d(np.asarray(keys, dtype=np.float64))
    m = FastOneClassSVM(C=1.0 / (nu * len(X)), ktype="r", kpar=kpar).seed_from_qp(X)
    return sorted(m.R)


def extract_global_layer_keys(model, input_ids, layer_idx: int, *, batch: int = 0,
                              head: int = 0) -> np.ndarray:
    """Run the model and return the SVDD key set (T, d) that grafted global layer
    ``layer_idx`` gates on, for one (batch, head). The keys are exactly what
    ``SVGlobalAttention._project_qkv`` produces (post q/k-norm, post-RoPE if enabled,
    GQA-expanded), so the exact solver sees the live model's keys. Requires execution
    (torch)."""
    import torch

    from .layer_select import find_global_attention_layers

    target = dict(find_global_attention_layers(model)).get(layer_idx)
    if target is None:
        raise ValueError(f"layer {layer_idx} is not a grafted global SV layer")

    cap: Dict[str, "torch.Tensor"] = {}

    def _pre_hook(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        pe = kwargs.get("position_embeddings")
        with torch.no_grad():
            _, k, _ = mod._project_qkv(hs, pe)
        cap["k"] = k.detach()

    h = target.self_attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        h.remove()
    # .cpu() BEFORE float64: MPS has no float64, so the cast must happen off-device.
    return cap["k"][batch, head].cpu().to(torch.float64).numpy()      # (T, d)

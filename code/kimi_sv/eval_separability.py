"""When is a KDA record static, transportable, or replay-dependent?

A fixed ingestion-time receipt is sufficient only when a record's state
contribution is unchanged by later writes. KDA does not have that static
property. It is nevertheless affine in the prior recurrent state when future
kernel inputs are held fixed, so a full transition transport can propagate a
receipt exactly. The stronger record-omitted counterfactual changes later
activations and therefore changes both transition and write terms; this
experiment measures that forcing and identifies where replay remains necessary.

We begin by computing the same record's state contribution under two futures:

    dS_X = S(prefix + record + suffix_X) - S(prefix + suffix_X)

If dS_A != dS_B, a *static* ingestion-time receipt is insufficient. We then
transport that receipt through every captured KDA transition and verify exact
agreement under frozen suffix inputs.

Two levels, one conclusion:

End-to-end. Four full-model prefills; compare dS across suffixes per KDA layer.
This includes representation entanglement (suffix activations differ when the
record is present upstream), so it upper-bounds what any state-side mechanism
must undo.

Isolated. The per-token kernel inputs (k, v, g, beta) are captured from both
record-present and record-omitted runs. The recurrence is re-run under three
write rules:

``separable``  S += beta * v k^T -- the SV-native ledger rule. Contribution is
               suffix-independent by construction; measured as a control.
``decay``      S = S * diag(g) + beta * v k^T. Raw contribution is suffix-
               dependent, but dividing by the suffix's cumulative decay aligns
               the two exactly: decay is diagonal and commutative, so a ledger
               plus one decay accumulator recovers decrement.
``kda``        decay plus the delta rule. The correction term beta*(v - S k)
               reads the state, so a static or decay-only correction fails.
               A full affine transition transports the receipt exactly while
               suffix inputs are fixed.

End-to-end forcing. Present and omitted runs produce different suffix inputs.
We decompose their final difference into the transported boundary receipt,
changed transition terms, and changed write terms. Replay recomputes all three.

The reference recurrence is validated against the Metal kernel's state on the
same inputs before anything is concluded from it.

Runs on the synthetic case file by default. With ``--source cds`` or
``--source notes`` the records come from credentialed local MIMIC files; the
report holds only norms and ratios, and is audited for source text before the
write regardless.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

import mlx_lm.models.kimi_linear as kimi_linear_module
from mlx_lm.models.gated_delta import compute_g, gated_delta_ops

from .protocol import (
    CASE_SOURCES,
    DEFAULT_MODEL,
    case_records,
    load_model,
    load_source_cases,
    preamble_record,
)
from .state import classify_layers

RECURRENT = 3  # index of the delta-rule state in a KDA layer's ArraysCache


def fnorm(a: mx.array) -> float:
    a = a.astype(mx.float32)
    return float(mx.sqrt(mx.sum(a * a)).item())


def rel_diff(a: mx.array, b: mx.array) -> float:
    denom = max(fnorm(a), fnorm(b), 1e-12)
    return fnorm(a.astype(mx.float32) - b.astype(mx.float32)) / denom


class KernelTap:
    """Capture per-token KDA kernel inputs by wrapping ``gated_delta_update``.

    KDA layers call the kernel once per forward, in layer order, so the call
    index within a forward identifies the layer.
    """

    def __init__(self):
        self.real = kimi_linear_module.gated_delta_update
        self.wanted: set = set()
        self.captured: Dict[int, Dict[str, mx.array]] = {}
        self.calls = 0

    def install(self):
        def wrapper(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
            idx = self.calls
            self.calls += 1
            if idx in self.wanted:
                grab = {
                    "q": q,
                    "k": k,
                    "v": v,
                    "g": compute_g(A_log, a, dt_bias),
                    "beta": mx.sigmoid(b),
                }
                mx.eval(*grab.values())
                self.captured[idx] = grab
            return self.real(
                q, k, v, a, b, A_log, dt_bias, state=state, mask=mask, use_kernel=use_kernel
            )

        kimi_linear_module.gated_delta_update = wrapper

    def uninstall(self):
        kimi_linear_module.gated_delta_update = self.real

    def reset(self, wanted: set):
        self.wanted = set(wanted)
        self.captured = {}
        self.calls = 0


def prefill_states(model: Any, tokens: mx.array, kda_layers: List[int]) -> List[mx.array]:
    """One full prefill; return each KDA layer's recurrent state."""
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    states = [cache[i][RECURRENT] for i in kda_layers]
    mx.eval(*states)
    return states


def take_tokens(arrays: Dict[str, mx.array], idx: List[int]) -> Dict[str, mx.array]:
    sel = mx.array(idx)
    return {name: arr[:, sel] for name, arr in arrays.items()}


def run_kda(
    inp: Dict[str, mx.array],
    state: Optional[mx.array] = None,
) -> mx.array:
    _, state = gated_delta_ops(
        inp["q"].astype(mx.float32),
        inp["k"].astype(mx.float32),
        inp["v"].astype(mx.float32),
        inp["g"].astype(mx.float32),
        inp["beta"].astype(mx.float32),
        state=None if state is None else state.astype(mx.float32),
    )
    mx.eval(state)
    return state


@mx.compile
def _kda_transition_step(k, g, beta, state):
    """Apply only KDA's linear transition to a state difference."""

    decayed = state * g[..., None, :]
    recalled = (decayed * k[..., None, :]).sum(axis=-1)
    correction = k[..., None, :] * (
        recalled * beta[..., None]
    )[..., None]
    return decayed - correction


@mx.compile
def _kda_write_term(k, v, beta):
    return k[..., None, :] * (v * beta[..., None])[..., None]


def transport_kda_state(
    state: mx.array,
    inp: Dict[str, mx.array],
) -> mx.array:
    """Transport ``state`` through captured KDA transitions, without writes."""

    k = inp["k"].astype(mx.float32)
    g = inp["g"].astype(mx.float32)
    beta = inp["beta"].astype(mx.float32)
    transported = state.astype(mx.float32)
    for t in range(k.shape[1]):
        transported = _kda_transition_step(
            k[:, t], g[:, t], beta[:, t], transported
        )
    mx.eval(transported)
    return transported


def transported_record_receipt(
    inp: Dict[str, mx.array],
    prefix_idx: List[int],
    victim_idx: List[int],
    suffix_idx: List[int],
) -> mx.array:
    """Propagate a record's boundary contribution through a frozen suffix."""

    prefix = take_tokens(inp, prefix_idx)
    victim = take_tokens(inp, victim_idx)
    suffix = take_tokens(inp, suffix_idx)
    prefix_state = run_kda(prefix)
    with_victim = run_kda(victim, state=prefix_state)
    receipt = with_victim - prefix_state
    return transport_kda_state(receipt, suffix)


def decompose_kda_suffix_difference(
    initial_present: mx.array,
    initial_omitted: mx.array,
    present_suffix: Dict[str, mx.array],
    omitted_suffix: Dict[str, mx.array],
) -> Dict[str, mx.array]:
    """Decompose a record-present/omitted difference over aligned suffixes.

    For ``S' = T_t(S) + B_t``, the exact recurrence is

    ``delta' = T_present(delta)
                + [T_present(S_omit) - T_omit(S_omit)]
                + [B_present - B_omit]``.

    The returned components sum to the final state difference.
    """

    n_present = int(present_suffix["k"].shape[1])
    n_omitted = int(omitted_suffix["k"].shape[1])
    if n_present != n_omitted:
        raise ValueError("present and omitted suffixes must be token-aligned")

    present = {
        name: value.astype(mx.float32)
        for name, value in present_suffix.items()
    }
    omitted = {
        name: value.astype(mx.float32)
        for name, value in omitted_suffix.items()
    }
    receipt = initial_present.astype(mx.float32) - initial_omitted.astype(
        mx.float32
    )
    transition_forcing = mx.zeros_like(receipt)
    write_forcing = mx.zeros_like(receipt)
    omitted_state = initial_omitted.astype(mx.float32)

    for t in range(n_present):
        kp, gp, bp = (
            present["k"][:, t],
            present["g"][:, t],
            present["beta"][:, t],
        )
        ko, go, bo = (
            omitted["k"][:, t],
            omitted["g"][:, t],
            omitted["beta"][:, t],
        )
        present_on_omitted = _kda_transition_step(
            kp, gp, bp, omitted_state
        )
        omitted_transition = _kda_transition_step(
            ko, go, bo, omitted_state
        )
        transition_gap = present_on_omitted - omitted_transition
        present_write = _kda_write_term(
            kp, present["v"][:, t], bp
        )
        omitted_write = _kda_write_term(
            ko, omitted["v"][:, t], bo
        )

        receipt = _kda_transition_step(kp, gp, bp, receipt)
        transition_forcing = _kda_transition_step(
            kp, gp, bp, transition_forcing
        ) + transition_gap
        write_forcing = _kda_transition_step(
            kp, gp, bp, write_forcing
        ) + present_write - omitted_write
        omitted_state = omitted_transition + omitted_write

    total = receipt + transition_forcing + write_forcing
    mx.eval(receipt, transition_forcing, write_forcing, total)
    return {
        "transported_receipt": receipt,
        "transition_forcing": transition_forcing,
        "write_forcing": write_forcing,
        "total": total,
    }


@mx.compile
def _decay_step(k, v, g, beta, state):
    state = state * g[..., None, :]
    return state + k[..., None, :] * (v * beta[..., None])[..., None]


@mx.compile
def _separable_step(k, v, beta, state):
    return state + k[..., None, :] * (v * beta[..., None])[..., None]


def run_rule(inp: Dict[str, mx.array], rule: str) -> mx.array:
    k = inp["k"].astype(mx.float32)
    v = inp["v"].astype(mx.float32)
    g = inp["g"].astype(mx.float32)
    beta = inp["beta"].astype(mx.float32)
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    state = mx.zeros((B, H, Dv, Dk), dtype=mx.float32)
    for t in range(T):
        if rule == "decay":
            state = _decay_step(k[:, t], v[:, t], g[:, t], beta[:, t], state)
        else:
            state = _separable_step(k[:, t], v[:, t], beta[:, t], state)
    mx.eval(state)
    return state


def suffix_decay(inp: Dict[str, mx.array], suffix_idx: List[int]) -> mx.array:
    """Cumulative per-channel decay over the suffix, for the ledger correction."""
    g = inp["g"].astype(mx.float32)
    sel = mx.array(suffix_idx)
    return mx.prod(g[:, sel], axis=1)  # (B, H, Dk)


def measure(
    model: Any,
    kda_layers: List[int],
    probe_layers: List[int],
    prefix: mx.array,
    victim: mx.array,
    suffix_a: mx.array,
    suffix_b: mx.array,
) -> Dict[str, Any]:
    """One victim/suffix-pair configuration: end-to-end and isolated metrics."""
    probe_calls = {kda_layers.index(i): i for i in probe_layers}  # call idx -> layer
    n_suffix = min(int(suffix_a.shape[1]), int(suffix_b.shape[1]))
    suffix_a = suffix_a[:, :n_suffix]
    suffix_b = suffix_b[:, :n_suffix]
    n_prefix, n_victim = int(prefix.shape[1]), int(victim.shape[1])

    def seq(*parts: mx.array) -> mx.array:
        return mx.concatenate(parts, axis=1)

    tap = KernelTap()
    tap.install()
    try:
        # -- end-to-end, capturing kernel inputs on the record-present runs --
        tap.reset(set(probe_calls))
        s_pra = prefill_states(model, seq(prefix, victim, suffix_a), kda_layers)
        cap_a = {probe_calls[c]: arrs for c, arrs in tap.captured.items()}

        tap.reset(set(probe_calls))
        s_prb = prefill_states(model, seq(prefix, victim, suffix_b), kda_layers)
        cap_b = {probe_calls[c]: arrs for c, arrs in tap.captured.items()}

        tap.reset(set(probe_calls))
        s_pa = prefill_states(model, seq(prefix, suffix_a), kda_layers)
        omit_a = {probe_calls[c]: arrs for c, arrs in tap.captured.items()}

        tap.reset(set(probe_calls))
        s_pb = prefill_states(model, seq(prefix, suffix_b), kda_layers)
        omit_b = {probe_calls[c]: arrs for c, arrs in tap.captured.items()}
    finally:
        tap.uninstall()

    e2e = {}
    for pos, layer in enumerate(kda_layers):
        d_a = s_pra[pos].astype(mx.float32) - s_pa[pos].astype(mx.float32)
        d_b = s_prb[pos].astype(mx.float32) - s_pb[pos].astype(mx.float32)
        e2e[str(layer)] = {
            "rel": rel_diff(d_a, d_b),
            "norm_dS_A": fnorm(d_a),
            "norm_dS_B": fnorm(d_b),
            "norm_S": fnorm(s_pra[pos]),
        }

    # -- isolated recurrence on identical captured inputs --------------------
    with_idx = list(range(n_prefix + n_victim + n_suffix))
    prefix_idx = list(range(n_prefix))
    victim_idx = list(range(n_prefix, n_prefix + n_victim))
    skip_idx = list(range(n_prefix)) + list(
        range(n_prefix + n_victim, n_prefix + n_victim + n_suffix)
    )
    suffix_idx = list(range(n_prefix + n_victim, n_prefix + n_victim + n_suffix))
    omitted_suffix_idx = list(range(n_prefix, n_prefix + n_suffix))

    isolated: Dict[str, Dict[str, float]] = {}
    forcing: Dict[str, Dict[str, Dict[str, float]]] = {}
    validation: Dict[str, float] = {}
    for layer in probe_layers:
        pos = kda_layers.index(layer)

        # The sequential reference must reproduce the Metal kernel's state
        # before its variants mean anything.
        ref_full = run_kda(cap_a[layer])
        validation[str(layer)] = rel_diff(ref_full, s_pra[pos])

        deltas: Dict[str, Dict[str, mx.array]] = {}
        decays: Dict[str, mx.array] = {}
        transport_errors: List[float] = []
        transported: Dict[str, mx.array] = {}
        layer_key = str(layer)
        forcing[layer_key] = {}
        for name, cap in (("A", cap_a[layer]), ("B", cap_b[layer])):
            d: Dict[str, mx.array] = {}
            d["kda"] = run_kda(take_tokens(cap, with_idx)) - run_kda(
                take_tokens(cap, skip_idx)
            )
            for rule in ("decay", "separable"):
                d[rule] = run_rule(take_tokens(cap, with_idx), rule) - run_rule(
                    take_tokens(cap, skip_idx), rule
                )
            deltas[name] = d
            decays[name] = suffix_decay(cap, suffix_idx)[..., None, :]
            transported[name] = transported_record_receipt(
                cap, prefix_idx, victim_idx, suffix_idx
            )
            transport_errors.append(rel_diff(d["kda"], transported[name]))

            omitted_cap = omit_a[layer] if name == "A" else omit_b[layer]
            initial_present = run_kda(
                take_tokens(cap, prefix_idx + victim_idx)
            )
            initial_omitted = run_kda(
                take_tokens(omitted_cap, prefix_idx)
            )
            components = decompose_kda_suffix_difference(
                initial_present,
                initial_omitted,
                take_tokens(cap, suffix_idx),
                take_tokens(omitted_cap, omitted_suffix_idx),
            )
            direct = run_kda(take_tokens(cap, with_idx)) - run_kda(
                take_tokens(
                    omitted_cap,
                    prefix_idx + omitted_suffix_idx,
                )
            )
            forcing[layer_key][name] = {
                "activation_rel_diff": {
                    field: rel_diff(
                        take_tokens(cap, suffix_idx)[field],
                        take_tokens(
                            omitted_cap, omitted_suffix_idx
                        )[field],
                    )
                    for field in ("q", "k", "v", "g", "beta")
                },
                "decomposition_error": rel_diff(
                    components["total"], direct
                ),
                "transported_receipt_norm": fnorm(
                    components["transported_receipt"]
                ),
                "transition_forcing_norm": fnorm(
                    components["transition_forcing"]
                ),
                "write_forcing_norm": fnorm(
                    components["write_forcing"]
                ),
                "direct_difference_norm": fnorm(direct),
                "forcing_fraction": fnorm(
                    components["transition_forcing"]
                    + components["write_forcing"]
                )
                / max(fnorm(direct), 1e-12),
            }

        out: Dict[str, float] = {
            rule: rel_diff(deltas["A"][rule], deltas["B"][rule])
            for rule in ("separable", "decay", "kda")
        }
        # A ledger correction may rescale by the suffix's cumulative decay.
        # Test dS_A/D_A = dS_B/D_B by cross-multiplication (no underflowing
        # division): decay-only must align exactly, kda must not.
        for rule in ("decay", "kda"):
            out[f"{rule}_corrected"] = rel_diff(
                deltas["A"][rule] * decays["B"], deltas["B"][rule] * decays["A"]
            )
        out["norm_dS_kda_A"] = fnorm(deltas["A"]["kda"])
        out["norm_dS_kda_B"] = fnorm(deltas["B"]["kda"])
        out["kda_transport_error_max"] = max(transport_errors)
        out["transported_receipt_rel_diff"] = rel_diff(
            transported["A"], transported["B"]
        )
        isolated[str(layer)] = out

    return {
        "tokens": {"prefix": n_prefix, "victim": n_victim, "suffix": n_suffix},
        "end_to_end_per_layer": e2e,
        "isolated_rel_diff": isolated,
        "end_to_end_forcing": forcing,
        "reference_vs_kernel_rel_diff": validation,
    }


def summarize(values: List[float]) -> Dict[str, float]:
    s = sorted(values)
    return {"min": s[0], "median": s[len(s) // 2], "max": s[-1]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--report-model-label",
        default=None,
        help="portable model identifier recorded instead of a local snapshot path",
    )
    ap.add_argument("--source", choices=CASE_SOURCES, default="synthetic")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--body-chars", type=int, default=None)
    ap.add_argument("--out", default="outputs/kimi_sv/kda_separability.json")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument(
        "--lazy-load",
        action="store_true",
        help="defer full parameter materialization until model execution",
    )
    ap.add_argument("--cache-limit-gb", type=float, default=2.0)
    args = ap.parse_args(argv)

    cases, _source_stats = load_source_cases(
        args.source,
        15,
        data_dir=args.data_dir,
        skip=args.skip,
        body_chars=args.body_chars,
    )

    mx.clear_cache()
    mx.set_cache_limit(int(args.cache_limit_gb * 1024 ** 3))
    t0 = time.perf_counter()
    model, tokenizer, label = load_model(
        args.model,
        tiny=args.tiny,
        lazy=args.lazy_load,
    )
    print(f"loaded {label} in {(time.perf_counter() - t0)/60:.1f} min", flush=True)

    kda_layers, _ = classify_layers(model)
    # First, middle, and last KDA layers carry the isolated study.
    probe_layers = sorted({kda_layers[0], kda_layers[len(kda_layers) // 2], kda_layers[-1]})
    records = [preamble_record(tokenizer)] + case_records(tokenizer, cases)
    toks = [r.tokens for r in records]
    prefix = mx.concatenate(toks[0:5], axis=1)  # preamble + cases 0-3
    pool = toks[8:16]  # cases 7-14 supply the two suffixes

    def cat(parts: List[mx.array]) -> mx.array:
        return mx.concatenate(parts, axis=1)

    pairings = {
        "halves": (cat(pool[:4]), cat(pool[4:])),
        "interleaved": (cat(pool[0::2]), cat(pool[1::2])),
    }
    configs = {
        f"victim{v}-{pname}": (toks[1 + v], pa, pb)
        for v in (4, 5, 6)  # cases 4-6 take turns as the victim
        for pname, (pa, pb) in pairings.items()
    }

    results: Dict[str, Any] = {}
    for name, (victim, sa, sb) in configs.items():
        res = measure(model, kda_layers, probe_layers, prefix, victim, sa, sb)
        results[name] = res
        e2e_rel = [v["rel"] for v in res["end_to_end_per_layer"].values()]
        iso = res["isolated_rel_diff"]
        print(
            f"{name}: e2e rel median {sorted(e2e_rel)[len(e2e_rel)//2]:.3f}; "
            + "; ".join(
                f"L{l} kda static {iso[l]['kda']:.3f}, "
                f"transport err {iso[l]['kda_transport_error_max']:.1e}, "
                f"decay {iso[l]['decay']:.3f}->corr {iso[l]['decay_corrected']:.1e}, "
                f"sep {iso[l]['separable']:.1e}"
                for l in sorted(iso, key=int)
            ),
            flush=True,
        )

    e2e_all = [
        v["rel"] for res in results.values() for v in res["end_to_end_per_layer"].values()
    ]
    survive_all = [
        v["norm_dS_A"] / max(v["norm_S"], 1e-12)
        for res in results.values()
        for v in res["end_to_end_per_layer"].values()
    ]
    iso_all: Dict[str, List[float]] = {}
    for res in results.values():
        for stats in res["isolated_rel_diff"].values():
            for rule in (
                "separable",
                "decay",
                "decay_corrected",
                "kda",
                "kda_corrected",
                "kda_transport_error_max",
                "transported_receipt_rel_diff",
            ):
                iso_all.setdefault(rule, []).append(stats[rule])
    forcing_all: Dict[str, List[float]] = {}
    activation_all: Dict[str, List[float]] = {}
    for res in results.values():
        for suffixes in res["end_to_end_forcing"].values():
            for stats in suffixes.values():
                for field, value in stats["activation_rel_diff"].items():
                    activation_all.setdefault(field, []).append(value)
                for metric in (
                    "decomposition_error",
                    "transported_receipt_norm",
                    "transition_forcing_norm",
                    "write_forcing_norm",
                    "direct_difference_norm",
                    "forcing_fraction",
                ):
                    forcing_all.setdefault(metric, []).append(stats[metric])
    val_all = [
        v for res in results.values() for v in res["reference_vs_kernel_rel_diff"].values()
    ]

    summary = {
        "end_to_end_rel": summarize(e2e_all),
        "surviving_contribution_norm_fraction": summarize(survive_all),
        "isolated": {rule: summarize(vals) for rule, vals in iso_all.items()},
        "end_to_end_forcing": {
            metric: summarize(vals)
            for metric, vals in forcing_all.items()
        },
        "suffix_activation_rel_diff": {
            field: summarize(vals)
            for field, vals in activation_all.items()
        },
        "reference_vs_kernel_max": max(val_all),
        "n_configs": len(results),
        "n_kda_layers": len(kda_layers),
        "isolated_layers": probe_layers,
    }
    print(
        f"summary over {len(results)} configs: e2e rel "
        f"{summary['end_to_end_rel']['min']:.3f}/"
        f"{summary['end_to_end_rel']['median']:.3f}/"
        f"{summary['end_to_end_rel']['max']:.3f} (min/med/max); isolated kda "
        f"{summary['isolated']['kda']['min']:.3f}-{summary['isolated']['kda']['max']:.3f} "
        f"-> full-transport error <= "
        f"{summary['isolated']['kda_transport_error_max']['max']:.1e}; "
        f"decay corrected "
        f"{summary['isolated']['decay_corrected']['max']:.1e}; separable "
        f"{summary['isolated']['separable']['max']:.1e}; "
        f"end-to-end decomposition error "
        f"{summary['end_to_end_forcing']['decomposition_error']['max']:.1e}; "
        f"reference vs kernel max {summary['reference_vs_kernel_max']:.1e}",
        flush=True,
    )

    report = {
        "model": args.report_model_label or label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "source": args.source,
        "contains_source_text": False,
        "protocol": {
            "kda_layers": kda_layers,
            "isolated_layers": probe_layers,
            "metric": "||dS_A - dS_B||_F / max(||dS_A||_F, ||dS_B||_F)",
            "receipt_contract": (
                "static and decay-only ledgers are compared with full affine "
                "transition transport; end-to-end differences are decomposed "
                "into transported receipt, transition forcing, and write forcing"
            ),
        },
        "summary": summary,
        "configs": results,
    }
    payload = json.dumps(report, indent=2)
    if args.source != "synthetic":
        from mimic import assert_no_source_text

        assert_no_source_text(payload, cases)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload)
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

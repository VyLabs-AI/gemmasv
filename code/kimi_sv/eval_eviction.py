"""Matched-budget eviction: certified SV support versus heavy-hitter selection.

The question the companion paper answered on synthetic data -- given a fixed
budget of context entries to keep, which selection rule preserves recall? --
transplanted onto Kimi Linear's global MLA layers under plain softmax.

One context of synthetic records is prefilled once with plain (ungated) blocked
attention, during which each grafted layer accumulates heavy-hitter scores: the
softmax mass each key position has received so far, the statistic H2O keeps by.
After prefill, each policy nominates, per layer, the positions to keep among the
prefix candidates:

``sv``         the support set of the layer's SVDD over its shared KV latent --
               the positions the certified gate would keep -- at unit weight;
``sv-recal``   the same, with the kernel bandwidth re-estimated over the full
               context instead of frozen at the first forward (diagnostic for
               the bandwidth heuristic);
``h2o``        the positions with the highest accumulated attention mass;
``h2o-recent`` H2O as conventionally deployed: half the budget always keeps
               the most recent candidates, half goes to heavy hitters;
``random-k``   a uniform draw, one per seed.

Two further arms run the gate itself rather than a mask, on the same context
and the same support sets:

``gated``       the SV readout as deployed -- prefix weights multiplied by the
                solved alphas and renormalized. Against ``sv`` this isolates
                the alpha *weighting* from support *membership*: both read the
                same kept set, one with solved weights, one at unit weight.
``gated-evict`` the gate plus point-in-time removal of its current reserves;
                later-admission equivalence is not claimed.

The budget is matched per layer: every masking policy keeps exactly as many
positions as the SV support set, so the only difference is *which* positions.
Eviction is masking -- dropped keys carry zero weight and the readout
renormalizes over the survivors, a softmax over the kept set. Positions at or
after the final chunk boundary stay resident under every policy. Recall of
every record is then probed teacher-forced plus a greedy exact-match answer,
without disturbing the memory.

Following the sweep's lesson, the comparator is resident-side recall (mean log
probability of each record's secret with the record in context), not lift.
Synthetic records only; no credentialed source is involved.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

from svattn.mlx_svdd import svdd_fista_mlx

from .graft import graft_sv_into_kimi, set_drop_positions
from .protocol import (
    DEFAULT_MODEL,
    answer_text,
    case_records,
    filler_record,
    generated_cases,
    load_model,
    preamble_record,
    probe_secret,
)
from .records import Record, RecordMemory
from .sv_mla_attention import median_bandwidth, rbf_gram

ALPHA_TOL = 1e-8


def sv_support(attn: Any, latent: mx.array, kpar: Optional[float] = None) -> List[int]:
    """Support positions of the SVDD the gate would solve at this boundary.

    Solved exactly as the live gate does -- same frozen bandwidth (unless
    ``kpar`` overrides it), same box, same iteration budget, same seed -- but
    over the plain-attention cache, so the policy is judged on the same
    representation H2O scored.
    """
    gate_keys = attn._gate_keys(latent)
    n = int(gate_keys.shape[1])
    gram = rbf_gram(gate_keys, gate_keys, attn.kpar if kpar is None else kpar)
    mx.random.seed(attn.solver_seed)
    alpha = svdd_fista_mlx(gram, attn._box(n), iters=attn.fista_iters)
    flags = (alpha[0] > ALPHA_TOL).tolist()
    return [i for i, hit in enumerate(flags) if hit]


def drop_fraction(drops: Optional[Dict[int, List[int]]], span: range) -> float:
    """Mean over grafted layers of the fraction of ``span`` that is evicted."""
    if not drops or len(span) == 0:
        return 0.0
    fractions = [
        len(set(layer_drops) & set(span)) / len(span)
        for layer_drops in drops.values()
    ]
    return sum(fractions) / len(fractions)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--records", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument(
        "--box-c",
        type=float,
        default=None,
        help="fixed SVDD box instead of 1/(nu*n); a length-independent box "
        "decouples per-token gate weight from context length",
    )
    ap.add_argument(
        "--preserve-mass",
        action="store_true",
        help="gated arms rescale the gated prefix to its pre-gate mass share, "
        "making the gate a pure redistribution within the prefix",
    )
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--greedy-tokens", type=int, default=12)
    ap.add_argument("--out", default="outputs/kimi_sv/eviction_matched_budget.json")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    model, tokenizer, label = load_model(args.model, tiny=args.tiny)
    print(f"loaded {label} in {(time.perf_counter() - t0)/60:.1f} min", flush=True)

    # gate=False: the blocked path runs plain softmax, so prefill, heavy-hitter
    # scores, and every arm's masked readout share one attention rule.
    grafted = graft_sv_into_kimi(
        model,
        mode="latent",
        nu=args.nu,
        C=args.box_c,
        chunk=args.chunk,
        gate=False,
        preserve_prefix_mass=args.preserve_mass,
    )
    for i in grafted:
        model.layers[i].self_attn.h2o_accumulate = True

    cases = generated_cases(args.records)
    records: List[Record] = [preamble_record(tokenizer)] + case_records(
        tokenizer, cases
    )
    body_tokens = sum(r.n_tokens for r in records)
    # Pad to a chunk boundary so every record sits strictly inside the
    # candidate region and the local window contains no record content.
    pad = (args.chunk - body_tokens % args.chunk) % args.chunk
    if pad:
        records.append(filler_record(tokenizer, pad, key="__pad__"))

    memory = RecordMemory(model)
    spans: Dict[str, range] = {}
    offset = 0
    t0 = time.perf_counter()
    for r in records:
        spans[r.key] = range(offset, offset + r.n_tokens)
        memory.ingest(r)
        offset += r.n_tokens
    print(f"ingested {offset} tokens in {time.perf_counter() - t0:.1f}s", flush=True)

    for i in grafted:
        model.layers[i].self_attn.h2o_accumulate = False

    total = memory.token_offset
    boundary = (total // args.chunk) * args.chunk
    last_case_end = max(spans[c["key"]].stop for c in cases)
    if last_case_end > boundary:
        raise AssertionError("a record extends past the candidate boundary")

    # -- per-layer keep sets at matched budget -----------------------------
    t0 = time.perf_counter()
    budgets: Dict[int, int] = {}
    overlap: Dict[int, float] = {}
    bos_kept: Dict[int, Dict[str, bool]] = {}
    policies: Dict[str, Dict[int, List[int]]] = {
        name: {} for name in ("sv", "sv-recal", "h2o", "h2o-recent")
    }
    for s in range(args.random_draws):
        policies[f"random-{s}"] = {}

    everything = set(range(boundary))
    for i in grafted:
        attn = model.layers[i].self_attn
        latent = memory.cache[i].state[0][:, :, :boundary, :]
        support = sv_support(attn, latent)
        k = len(support)
        budgets[i] = k
        policies["sv"][i] = sorted(everything - set(support))

        recal = median_bandwidth(attn._gate_keys(latent))
        support_recal = sv_support(attn, latent, kpar=recal)
        policies["sv-recal"][i] = sorted(everything - set(support_recal))

        scores = attn.h2o_scores[:boundary].tolist()
        order = sorted(range(boundary), key=lambda p: scores[p])
        policies["h2o"][i] = sorted(order[: boundary - k])
        keep_h2o = set(order[boundary - k :])
        overlap[i] = len(keep_h2o & set(support)) / max(k, 1)

        # Conventional H2O splits its budget: the most recent candidates are
        # always resident, heavy hitters fill the remainder.
        recent = set(range(boundary - k // 2, boundary))
        heavy = [p for p in reversed(order) if p not in recent][: k - len(recent)]
        policies["h2o-recent"][i] = sorted(everything - recent - set(heavy))

        bos_kept[i] = {"sv": 0 in support, "h2o": 0 in keep_h2o}

        for s in range(args.random_draws):
            rng = random.Random(9973 * s + i)
            policies[f"random-{s}"][i] = sorted(
                rng.sample(range(boundary), boundary - k)
            )
    print(
        f"keep sets solved in {time.perf_counter() - t0:.1f}s; "
        f"budget {min(budgets.values())}-{max(budgets.values())} of {boundary} "
        f"per layer, SV/H2O keep overlap "
        f"{min(overlap.values()):.2f}-{max(overlap.values()):.2f}",
        flush=True,
    )

    # -- probe every record under every arm ---------------------------------
    arms = ["none", "gated", "gated-evict", "sv", "sv-recal", "h2o", "h2o-recent"]
    arms += [f"random-{s}" for s in range(args.random_draws)]
    results: Dict[str, Any] = {}
    for arm in arms:
        if arm == "none" or arm == "gated":
            drops = None
        elif arm == "gated-evict":
            drops = policies["sv"]
        else:
            drops = policies[arm]
        for i in grafted:
            model.layers[i].self_attn.gate = arm.startswith("gated")
        set_drop_positions(model, drops)
        t0 = time.perf_counter()

        per_record: Dict[str, Any] = {}
        for case in cases:
            stats = probe_secret(memory, tokenizer, case)
            answer = answer_text(
                memory, tokenizer, case["question"], max_tokens=args.greedy_tokens
            )
            per_record[case["key"]] = {
                "span": [spans[case["key"]].start, spans[case["key"]].stop],
                "mean_log_probability": stats.mean_log_probability,
                "first_token_rank": stats.first_token_rank,
                "greedy_exact": answer.strip().startswith(case["secret"].strip()),
                "record_fraction_evicted": drop_fraction(
                    drops, spans[case["key"]]
                ),
            }

        recalls = [r["mean_log_probability"] for r in per_record.values()]
        ranks = [r["first_token_rank"] for r in per_record.values()]
        results[arm] = {
            "per_record": per_record,
            "aggregate": {
                "recall_mean": statistics.mean(recalls),
                "recall_min": min(recalls),
                "rank_1_count": sum(1 for r in ranks if r == 1),
                "rank_worst": max(ranks),
                "greedy_exact_count": sum(
                    1 for r in per_record.values() if r["greedy_exact"]
                ),
            },
            "seconds": time.perf_counter() - t0,
        }
        agg = results[arm]["aggregate"]
        print(
            f"{arm:>10}: recall {agg['recall_mean']:+.3f} "
            f"(min {agg['recall_min']:+.3f}), rank1 {agg['rank_1_count']}/{len(cases)} "
            f"(worst {agg['rank_worst']}), greedy exact "
            f"{agg['greedy_exact_count']}/{len(cases)} "
            f"[{results[arm]['seconds']:.0f}s]",
            flush=True,
        )

    set_drop_positions(model, None)
    for i in grafted:
        model.layers[i].self_attn.gate = False

    report = {
        "model": label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "protocol": {
            "records": len(cases),
            "context_tokens": total,
            "candidate_boundary": boundary,
            "chunk": args.chunk,
            "nu": args.nu,
            "box_c": args.box_c,
            "preserve_prefix_mass": args.preserve_mass,
            "greedy_tokens": args.greedy_tokens,
            "budget_matched": (
                "masking arms keep exactly the SV support count per layer; "
                "sv-recal keeps its own support count (diagnostic)"
            ),
        },
        "per_layer": {
            str(i): {
                "kept": budgets[i],
                "keep_fraction": budgets[i] / boundary,
                "sv_h2o_keep_overlap": overlap[i],
                "bos_kept": bos_kept[i],
                "kept_by_policy": {
                    name: boundary - len(drops[i])
                    for name, drops in policies.items()
                },
            }
            for i in grafted
        },
        "arms": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

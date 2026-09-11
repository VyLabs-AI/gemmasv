"""Verification of replay deletion on the released Kimi Linear weights.

Ingests synthetic records into one persistent context, measures the readout
advantage each record confers, deletes one record by rolling the hybrid cache
back and replaying the survivors, and compares the result against a context that
never ingested the record at all.

Nothing here is a language-quality claim. The numbers that matter are the readout
advantage before deletion (the record must have mattered), the residual
disagreement against the never-ingested reference (should be at or near zero),
and the cost of replay versus rebuilding the context.

Usage::

    HF_HUB_CACHE=/path/to/huggingface/cache \\
      .venv311/bin/python -m kimi_sv.verify_real --victim 1
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

from .graft import gate_stats, graft_sv_into_kimi
from .probes import lift_nats
from .protocol import (
    CASES,
    DEFAULT_MODEL,
    answer_text,
    case_records,
    encode,
    generated_cases,
    load_model,
    preamble_record,
    score_all,
)
from .records import RecordMemory, max_abs_diff, state_max_abs_diff
from .state import describe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--victim", type=int, default=1, help="index into the case list")
    ap.add_argument(
        "--records",
        type=int,
        default=len(CASES),
        help="cases beyond the hand-written four are generated synthetics",
    )
    ap.add_argument(
        "--out", default="outputs/kimi_sv/verify_real.json", help="JSON report path"
    )
    ap.add_argument(
        "--tiny",
        action="store_true",
        help="dry-run every code path on a miniature random model (no weights)",
    )
    ap.add_argument(
        "--graft",
        choices=["none", "latent", "per_head", "shuffled"],
        default="none",
        help="replace softmax on the global MLA layers with the certified SV gate",
    )
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="gate granularity; boundaries below this never gate, so a short "
        "context needs a small chunk for the gate to engage at all",
    )
    ap.add_argument(
        "--solver-seed",
        type=int,
        default=0,
        help="seeds the FISTA power iteration and, for --graft shuffled, the "
        "permutation draw; vary it to sample fresh random-eviction controls",
    )
    ap.add_argument(
        "--preserve-mass",
        action="store_true",
        help="rescale the gated prefix to its pre-gate mass share so the gate "
        "redistributes within the prefix instead of crushing it",
    )
    args = ap.parse_args()

    cases = generated_cases(args.records)
    if not 0 <= args.victim < len(cases):
        raise SystemExit(f"--victim must be in [0, {len(cases)})")
    victim_key = cases[args.victim]["key"]

    t0 = time.perf_counter()
    model, tokenizer, label = load_model(args.model, tiny=args.tiny)
    load_seconds = time.perf_counter() - t0
    print(f"loaded {label} in {load_seconds/60:.1f} min", flush=True)

    grafted: List[int] = []
    if args.graft != "none":
        grafted = graft_sv_into_kimi(
            model,
            mode=args.graft,
            nu=args.nu,
            chunk=args.chunk,
            preserve_prefix_mass=args.preserve_mass,
            collect_stats=True,
            solver_seed=args.solver_seed,
        )
        label = f"{label} [SV graft: {args.graft}, nu={args.nu}, chunk={args.chunk}]"
        print(f"grafted SV gate onto global layers {grafted}", flush=True)

    preamble = preamble_record(tokenizer)
    records = case_records(tokenizer, cases)
    survivors = [r for r in records if r.key != victim_key]

    # Present: the full context, every record resident.
    present = RecordMemory(model)
    present.ingest(preamble)
    t0 = time.perf_counter()
    present.ingest_all(records)
    ingest_seconds = time.perf_counter() - t0
    cache_info = describe(model, present.cache)
    print(f"ingested {cache_info['token_offset']} tokens in {ingest_seconds:.1f}s")

    # Reference: a context that never saw the victim.
    reference = RecordMemory(model)
    reference.ingest(preamble)
    t0 = time.perf_counter()
    reference.ingest_all(survivors)
    rebuild_seconds = time.perf_counter() - t0

    present_stats = score_all(present, tokenizer, cases)
    never_stats = score_all(reference, tokenizer, cases)

    victim_question = cases[args.victim]["question"]
    behavior = {
        "question": victim_question,
        "with_record": answer_text(present, tokenizer, victim_question),
        "never_ingested": answer_text(reference, tokenizer, victim_question),
    }

    report = present.delete(victim_key)
    deleted_stats = score_all(present, tokenizer, cases)
    behavior["after_deletion"] = answer_text(present, tokenizer, victim_question)

    victim_probe = encode(tokenizer, "\n" + victim_question)
    logit_delta = max_abs_diff(
        present.probe(victim_probe), reference.probe(victim_probe)
    )
    state_delta = state_max_abs_diff(present, reference)

    result: Dict[str, Any] = {
        "model": label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "context": {
            "records": [r.key for r in records],
            "tokens": cache_info["token_offset"],
            "kda_layers": cache_info["kda_layers"],
            "mla_layers": cache_info["mla_layers"],
            "kda_bytes": cache_info["kda_bytes"],
            "mla_bytes": cache_info["mla_bytes"],
        },
        "victim": victim_key,
        "timing_seconds": {
            "load": load_seconds,
            "ingest_full": ingest_seconds,
            "rebuild_without_victim": rebuild_seconds,
            "delete_by_replay": report.seconds,
        },
        "deletion": {
            "rolled_back_to_token": report.rolled_back_to,
            "replayed_records": report.replayed_records,
            "replayed_tokens": report.replayed_tokens,
            "checkpoint_bytes": report.checkpoint_bytes,
        },
        "equivalence": {
            "max_abs_logit_delta_vs_never_ingested": logit_delta,
            "max_abs_kda_state_delta_vs_never_ingested": state_delta,
        },
        "behavior": behavior,
        "graft": {
            "mode": args.graft,
            "layers": grafted,
            "nu": args.nu,
            "chunk": args.chunk,
            "solver_seed": args.solver_seed,
            "preserve_prefix_mass": args.preserve_mass,
            "gate_stats": gate_stats(model) if grafted else {},
        },
        "probes": {},
    }

    # The reference context differs from the full one by exactly the victim, so
    # for the victim these deltas measure efficacy (did deletion remove it) and
    # for every other record they measure specificity (was it left alone).
    for key in [c["key"] for c in cases]:
        p, n, d = present_stats[key], never_stats[key], deleted_stats[key]
        result["probes"][key] = {
            "is_victim": key == victim_key,
            "role": "efficacy" if key == victim_key else "specificity",
            "present": p.as_dict(),
            "reference_without_victim": n.as_dict(),
            "after_deletion": d.as_dict(),
            "nats_vs_reference_before_deletion": lift_nats(p, n),
            "nats_vs_reference_after_deletion": lift_nats(d, n),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nvictim: {victim_key}")
    for key, row in result["probes"].items():
        print(
            f"  [{row['role']:11s}] {key}:"
            f" {row['nats_vs_reference_before_deletion']:+.3f} nats vs reference"
            f" -> {row['nats_vs_reference_after_deletion']:+.3e} after deletion"
            f" (rank {row['present']['first_token_rank']}"
            f" -> {row['after_deletion']['first_token_rank']})"
        )

    print(f"\nbehavioral answer to {behavior['question']!r}")
    for key in ("with_record", "never_ingested", "after_deletion"):
        print(f"  {key:16s} {behavior[key]!r}")
    print(
        f"\nequivalence vs never-ingested: logits {logit_delta:.3e},"
        f" KDA state {state_delta:.3e}"
    )
    print(
        f"delete-by-replay {report.seconds:.2f}s"
        f" ({report.replayed_tokens} tokens replayed)"
        f" vs rebuild {rebuild_seconds:.2f}s"
    )
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

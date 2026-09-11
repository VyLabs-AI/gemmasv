"""Measure reverse-decrement fallback versus nu on identical frozen 4B keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_sv.demo_server.certificate import certificate_overrides
from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_whole_record_unlearning import _record_memory
from gemma_sv.recovery_protocol import write_json_atomic


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "benchmarks" / "fallback_density_diagnostic_v1.json"
TOFU_REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"


def main(argv=None) -> int:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    manifest = json.loads(
        (ROOT / "benchmarks" / config["manifest"]).read_text()
    )
    spec = manifest["records"][int(config["manifest_index"])]
    model = config["model"]
    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=model["id"],
            model_revision=model["revision"],
            lora_path=None,
            device=args.device,
            dtype="float64",
            generation_tokens=1,
            window=1024,
            nu=0.7,
            preserve_prefix_mass=True,
            per_boundary_box=True,
            solver_seed=0,
        )
    )
    runtime.ensure_loaded()
    forget_rows = list(
        load_dataset(
            "locuslab/TOFU",
            "forget10",
            split="train",
            revision=TOFU_REVISION,
        )
    )
    retain_rows = list(
        load_dataset(
            "locuslab/TOFU",
            "retain90",
            split="train",
            revision=TOFU_REVISION,
        )
    )
    selected = [forget_rows[int(index)] for index in spec["forget_indices"]]
    retained = retain_rows[int(spec["retain_index"])]
    fillers = [str(row["answer"]) for row in retain_rows[:32]]
    present, _, _ = _record_memory(
        runtime,
        selected,
        retained,
        fillers,
        window=1024,
        n_fill=22,
        prefix_fillers=11,
    )
    memory = runtime.prefill_persistent(present.token_ids)
    keys = {
        layer_id: session.kf.detach().cpu().double().numpy()
        for layer_id, session in memory.layer_sessions.items()
    }
    kpars = {
        layer_id: float(session.kpar)
        for layer_id, session in memory.layer_sessions.items()
    }
    layer_ids = sorted(keys)
    heads = keys[layer_ids[0]].shape[0]
    chunk = int(runtime.layers[layer_ids[0]].self_attn.chunk)
    rows = []
    for nu in config["counterfactual_nu"]:
        print(f"nu={nu}", flush=True)
        result = certificate_overrides(
            keys,
            layer_ids,
            present.positions["forget"],
            memory.token_count,
            heads,
            nu=float(nu),
            chunk=chunk,
            per_boundary_box=True,
            kpar_by_layer=kpars,
        )
        rows.append(
            {
                "nu": float(nu),
                "decrement_fallbacks": result["n_fallback"],
                "affected_head_decrements": result["n_solves"],
                "head_gates": result["n_head_gates"],
                "fallback_fraction": result["n_fallback"]
                / max(result["n_solves"], 1),
                "partition_diagnostics": result["partition_diagnostics"],
                "max_functional_deviation": result[
                    "max_functional_deviation"
                ],
                "max_candidate_deviation": result["max_candidate_deviation"],
            }
        )
    report = {
        "schema": "gemma-sv-fallback-density-diagnostic-v1",
        "config": config,
        "memory_tokens": memory.token_count,
        "deleted_positions": len(present.positions["forget"]),
        "layers": layer_ids,
        "heads": heads,
        "rows": rows,
    }
    write_json_atomic(Path(args.out), report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

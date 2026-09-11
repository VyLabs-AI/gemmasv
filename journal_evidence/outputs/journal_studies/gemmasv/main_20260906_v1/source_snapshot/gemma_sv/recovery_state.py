"""Persist and restore the non-PEFT state learned during Gemma recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from gemma_sv.layer_select import find_global_attention_layers
from gemma_sv.recovery_protocol import write_json_atomic


STATE_FILENAME = "sv_recovery_state.json"
STATE_SCHEMA = "gemma-sv-recovery-state-v1"


def collect_recovery_state(
    model,
    *,
    model_id: str,
    model_revision: str | None,
) -> dict[str, Any]:
    """Collect learned per-layer bandwidths omitted by PEFT adapter saves."""

    layers: dict[str, dict[str, Any]] = {}
    mass_modes: set[bool] = set()
    box_modes: set[bool] = set()
    solver_seeds: set[int] = set()
    for layer_id, decoder_layer in find_global_attention_layers(model):
        attention = decoder_layer.self_attn
        if hasattr(attention, "log_kpar"):
            kpar = float(attention.log_kpar.detach().float().cpu().exp().item())
            source = "learned_log_kpar"
        elif getattr(attention, "kpar", None) is not None:
            kpar = float(attention.kpar)
            source = "fixed_kpar"
        else:
            raise RuntimeError(
                f"global layer {layer_id} has no fixed recovery bandwidth"
            )
        if not math.isfinite(kpar) or kpar <= 0:
            raise RuntimeError(
                f"global layer {layer_id} has invalid recovery bandwidth {kpar!r}"
            )
        preserve_prefix_mass = bool(
            getattr(attention, "preserve_prefix_mass", False)
        )
        mass_modes.add(preserve_prefix_mass)
        per_boundary_box = bool(
            getattr(attention, "per_boundary_box", False)
        )
        box_modes.add(per_boundary_box)
        solver_seed = int(getattr(attention, "solver_seed", 0))
        solver_seeds.add(solver_seed)
        layers[str(layer_id)] = {
            "kpar": kpar,
            "source": source,
            "chunk": int(attention.chunk),
            "nu": float(attention.nu),
            "readout": str(attention.readout),
            "preserve_prefix_mass": preserve_prefix_mass,
            "per_boundary_box": per_boundary_box,
            "solver_seed": solver_seed,
        }
    if not layers:
        raise RuntimeError("no global layers found while collecting recovery state")
    if len(mass_modes) != 1:
        raise RuntimeError("global layers disagree on prefix-mass preservation")
    if len(box_modes) != 1:
        raise RuntimeError("global layers disagree on boundary-box mode")
    if len(solver_seeds) != 1:
        raise RuntimeError("global layers disagree on solver seed")
    return {
        "schema": STATE_SCHEMA,
        "model": model_id,
        "model_revision": model_revision,
        "preserve_prefix_mass": mass_modes.pop(),
        "per_boundary_box": box_modes.pop(),
        "solver_seed": solver_seeds.pop(),
        "layers": layers,
    }


def save_recovery_state(adapter_path: str | Path, state: dict[str, Any]) -> Path:
    destination = Path(adapter_path) / STATE_FILENAME
    write_json_atomic(destination, state)
    return destination


def load_recovery_state(adapter_path: str | Path) -> dict[str, Any] | None:
    path = Path(adapter_path) / STATE_FILENAME
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"unsupported recovery state schema in {path}")
    if not isinstance(state.get("layers"), dict) or not state["layers"]:
        raise ValueError(f"recovery state has no layer bandwidths: {path}")
    return state


def apply_recovery_state(
    model,
    adapter_path: str | Path,
    *,
    strict: bool = True,
    allow_readout_override: bool = False,
) -> dict[str, Any] | None:
    """Apply saved bandwidths to an already grafted model."""

    state = load_recovery_state(adapter_path)
    if state is None:
        if strict:
            raise FileNotFoundError(Path(adapter_path) / STATE_FILENAME)
        return None
    expected = state["layers"]
    seen: set[str] = set()
    for layer_id, decoder_layer in find_global_attention_layers(model):
        key = str(layer_id)
        if key not in expected:
            if strict:
                raise ValueError(f"recovery state is missing global layer {key}")
            continue
        entry = expected[key]
        attention = decoder_layer.self_attn
        if strict:
            state_mass_mode = bool(state.get("preserve_prefix_mass", False))
            state_box_mode = bool(state.get("per_boundary_box", False))
            state_solver_seed = int(state.get("solver_seed", 0))
            for field, actual in (
                ("chunk", int(attention.chunk)),
                ("nu", float(attention.nu)),
                ("readout", str(attention.readout)),
                (
                    "preserve_prefix_mass",
                    bool(getattr(attention, "preserve_prefix_mass", False)),
                ),
                (
                    "per_boundary_box",
                    bool(getattr(attention, "per_boundary_box", False)),
                ),
                ("solver_seed", int(getattr(attention, "solver_seed", 0))),
            ):
                if field == "preserve_prefix_mass" and allow_readout_override:
                    continue
                recorded = (
                    entry.get(field, state_mass_mode)
                    if field == "preserve_prefix_mass"
                    else (
                        entry.get(field, state_box_mode)
                        if field == "per_boundary_box"
                        else (
                            entry.get(field, state_solver_seed)
                            if field == "solver_seed"
                            else entry.get(field)
                        )
                    )
                )
                if recorded != actual:
                    raise ValueError(
                        f"layer {key} {field} mismatch: "
                        f"{recorded!r} != {actual!r}"
                    )
        attention.kpar = float(entry["kpar"])
        attention._learn_kpar = False
        seen.add(key)
    missing = sorted(set(expected) - seen)
    if strict and missing:
        raise ValueError(f"recovery state has unmatched global layers: {missing}")
    return state

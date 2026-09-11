"""Pure helpers for persistent Gemma deletion-baseline evaluations.

The model runner lives in :mod:`gemma_sv.eval_persistent_deletion_baselines`.
This module intentionally keeps model loading out of its import path so context
construction, cache surgery, storage accounting, and report shaping can be
tested without downloading model weights.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DECAY_FACTOR = 0.01
ICUL_CORRECT_DEMONSTRATIONS = 4


@dataclass(frozen=True)
class PersistentDeletionContext:
    """One fixed packed context and its literal edited-context counterpart."""

    original_token_ids: tuple[int, ...]
    edited_token_ids: tuple[int, ...]
    forget_positions: tuple[int, ...]
    deletion_ranges: tuple[tuple[int, int], ...]
    retain_positions: tuple[int, ...]
    edited_retain_positions: tuple[int, ...]
    original_digest: str
    edited_digest: str


@dataclass(frozen=True)
class ICULContext:
    """Faithful ICUL prefix: one wrong answer plus correct demonstrations."""

    text: str
    wrong_answer_record_id: str
    demonstration_record_ids: tuple[str, ...]
    seed: int


def normalize_positions(
    positions: Iterable[int],
    *,
    total_tokens: int | None = None,
    require_nonempty: bool = True,
) -> tuple[int, ...]:
    """Return a sorted, unique, validated position tuple."""

    normalized = tuple(sorted({int(position) for position in positions}))
    if require_nonempty and not normalized:
        raise ValueError("at least one deletion position is required")
    if normalized and normalized[0] < 0:
        raise ValueError("deletion positions must be non-negative")
    if (
        total_tokens is not None
        and normalized
        and normalized[-1] >= int(total_tokens)
    ):
        raise ValueError("deletion position is outside the token context")
    return normalized


def contiguous_ranges(positions: Iterable[int]) -> tuple[tuple[int, int], ...]:
    """Convert positions to half-open contiguous ranges."""

    values = normalize_positions(positions)
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for position in values[1:]:
        if position != previous + 1:
            ranges.append((start, previous + 1))
            start = position
        previous = position
    ranges.append((start, previous + 1))
    return tuple(ranges)


def remove_token_positions(
    token_ids: Sequence[int],
    positions: Iterable[int],
) -> tuple[int, ...]:
    """Delete token positions exactly, preserving every retained token."""

    forget = normalize_positions(positions, total_tokens=len(token_ids))
    forgotten = set(forget)
    return tuple(
        int(token_id)
        for index, token_id in enumerate(token_ids)
        if index not in forgotten
    )


def shift_positions(
    positions: Iterable[int],
    deleted_positions: Iterable[int],
) -> tuple[int, ...]:
    """Map retained absolute positions through a delete-and-shift edit."""

    deleted = normalize_positions(deleted_positions)
    deleted_set = set(deleted)
    shifted = []
    for raw_position in positions:
        position = int(raw_position)
        if position in deleted_set:
            raise ValueError("cannot shift a position that was deleted")
        shift = sum(deleted_position < position for deleted_position in deleted)
        shifted.append(position - shift)
    return tuple(shifted)


def token_ids_digest(token_ids: Sequence[int]) -> str:
    """Hash token IDs using the persistent engine's int64 representation."""

    values = np.asarray([int(token_id) for token_id in token_ids], dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_synthetic_record_context(
    tokenizer,
    spec: Mapping[str, Any],
    fillers: Sequence[str],
    *,
    window: int,
    min_fillers: int,
    prefix_fillers: int,
    copies: int,
) -> PersistentDeletionContext:
    """Pack a target record, its paired retained record, and exact edit.

    The original context is built once.  The repack reference removes the
    target-record token union from that exact sequence; it does not substitute
    filler tokens or independently repack a length-matched counterfactual.
    """

    if copies < 1:
        raise ValueError("copies must be positive")
    if not fillers:
        raise ValueError("at least one filler is required")

    from gemma_sv.eval_robust_unlearning import pack_memory

    forget_names = [f"forget_copy_{index}" for index in range(copies)]
    target_records = [
        (
            name,
            str(spec["question"]),
            str(spec["answer"]),
        )
        for name in forget_names
    ]
    retain_record = (
        "retain",
        str(spec["retain_question"]),
        str(spec["retain_answer"]),
    )
    # Preserve the established retain-then-target ordering (important for the
    # first fixed-C boundary), but put one neutral record between target copies.
    # The deletion is therefore a genuine disjoint-span request rather than two
    # adjacent logical records that accidentally collapse to one token interval.
    records = [retain_record, target_records[0]]
    for index, target_record in enumerate(target_records[1:]):
        records.append(
            (
                f"copy_separator_{index}",
                "What neutral background note is retained?",
                str(fillers[index % len(fillers)]),
            )
        )
        records.append(target_record)
    packed = pack_memory(
        tokenizer,
        records,
        list(fillers),
        window=int(window),
        min_fillers=int(min_fillers),
        full_record_names=set(forget_names),
        prefix_fillers=int(prefix_fillers),
    )
    forget = normalize_positions(
        (
            position
            for name in forget_names
            for position in packed.positions[name]
        ),
        total_tokens=len(packed.token_ids),
    )
    retain = tuple(int(position) for position in packed.positions["retain"])
    edited = remove_token_positions(packed.token_ids, forget)
    return PersistentDeletionContext(
        original_token_ids=tuple(int(token_id) for token_id in packed.token_ids),
        edited_token_ids=edited,
        forget_positions=forget,
        deletion_ranges=contiguous_ranges(forget),
        retain_positions=retain,
        edited_retain_positions=shift_positions(retain, forget),
        original_digest=token_ids_digest(packed.token_ids),
        edited_digest=token_ids_digest(edited),
    )


def _stable_order_key(seed: int, target_id: str, candidate_id: str, kind: str):
    payload = f"{int(seed)}\0{target_id}\0{candidate_id}\0{kind}".encode()
    return hashlib.sha256(payload).digest()


def build_faithful_icul_context(
    target: Mapping[str, Any],
    all_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    correct_demonstrations: int = ICUL_CORRECT_DEMONSTRATIONS,
) -> ICULContext:
    """Construct ICUL with a wrong target answer and correct demonstrations.

    This follows the published three-step ICUL template: pair the forget input
    with a different answer, append correctly answered examples, then let the
    caller append the query.  Selection is deterministic for ``seed``.
    """

    if correct_demonstrations < 1:
        raise ValueError("ICUL requires at least one correct demonstration")
    target_id = str(target["record_id"])
    alternatives = [
        record
        for record in all_records
        if str(record["record_id"]) != target_id
        and str(record["answer"]) != str(target["answer"])
    ]
    if not alternatives:
        raise ValueError("ICUL requires a record with a different answer")
    wrong_source = min(
        alternatives,
        key=lambda record: _stable_order_key(
            seed,
            target_id,
            str(record["record_id"]),
            "wrong",
        ),
    )

    demonstration_candidates = [
        record
        for record in all_records
        if str(record["record_id"]) != target_id
    ]
    if len(demonstration_candidates) < correct_demonstrations:
        raise ValueError("not enough distinct retained ICUL demonstrations")
    demonstrations = sorted(
        demonstration_candidates,
        key=lambda record: _stable_order_key(
            seed,
            target_id,
            str(record["record_id"]),
            "correct",
        ),
    )[:correct_demonstrations]

    pieces = [
        "\n\nIn-context answer examples:\n",
        f"Question: {target['question']}\n",
        f"Answer: {wrong_source['answer']}\n",
    ]
    for record in demonstrations:
        pieces.extend(
            (
                f"Question: {record['retain_question']}\n",
                f"Answer: {record['retain_answer']}\n",
            )
        )
    return ICULContext(
        text="".join(pieces),
        wrong_answer_record_id=str(wrong_source["record_id"]),
        demonstration_record_ids=tuple(
            str(record["record_id"]) for record in demonstrations
        ),
        seed=int(seed),
    )


def full_vocabulary_kl(
    reference_log_probabilities: Sequence[float],
    method_log_probabilities: Sequence[float],
) -> float:
    """Compute ``KL(reference || method)`` over the full vocabulary."""

    log_p = np.asarray(reference_log_probabilities, dtype=np.float64)
    log_q = np.asarray(method_log_probabilities, dtype=np.float64)
    if log_p.shape != log_q.shape or log_p.ndim != 1:
        raise ValueError("log-probability vectors must be equal one-dimensional shapes")
    probability = np.exp(log_p)
    positive = probability > 0
    value = float(
        np.sum(probability[positive] * (log_p[positive] - log_q[positive]))
    )
    return max(0.0, value)


def first_token_rank(log_probabilities: Sequence[float], token_id: int) -> int:
    """One-indexed rank of a target token in a full-vocabulary distribution."""

    values = np.asarray(log_probabilities, dtype=np.float64)
    target = int(token_id)
    if values.ndim != 1 or not 0 <= target < values.shape[0]:
        raise ValueError("target token is outside the vocabulary distribution")
    return int(np.count_nonzero(values > values[target]) + 1)


def _torch_storage(tensor) -> tuple[tuple[Any, ...], int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    storage = tensor.untyped_storage()
    identity = getattr(storage, "_cdata", None)
    if identity is None:
        identity = storage.data_ptr() or id(storage)
    size = int(storage.nbytes())
    return ("torch", str(tensor.device), int(identity)), size


def _numpy_storage(array) -> tuple[tuple[Any, ...], int] | None:
    if not isinstance(array, np.ndarray):
        return None
    owner = array
    seen: set[int] = set()
    while isinstance(owner.base, np.ndarray) and id(owner.base) not in seen:
        seen.add(id(owner))
        owner = owner.base
    pointer = int(owner.__array_interface__["data"][0])
    return ("numpy", pointer, id(owner)), int(owner.nbytes)


def tensor_storage_report(root: Any) -> dict[str, int]:
    """Count unique underlying Torch/NumPy storages reachable from ``root``."""

    seen_objects: set[int] = set()
    storages: dict[tuple[Any, ...], tuple[str, int]] = {}

    def visit(value: Any) -> None:
        torch_storage = _torch_storage(value)
        if torch_storage is not None:
            key, size = torch_storage
            storages.setdefault(key, ("torch", size))
            return
        numpy_storage = _numpy_storage(value)
        if numpy_storage is not None:
            key, size = numpy_storage
            storages.setdefault(key, ("numpy", size))
            return
        if value is None or isinstance(
            value,
            (str, bytes, bytearray, int, float, bool, complex, ModuleType, type),
        ):
            return

        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)

        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for item in attributes.values():
                visit(item)
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if hasattr(value, slot):
                visit(getattr(value, slot))

    visit(root)
    torch_bytes = sum(
        size for backend, size in storages.values() if backend == "torch"
    )
    numpy_bytes = sum(
        size for backend, size in storages.values() if backend == "numpy"
    )
    return {
        "deduplicated_tensor_storage_bytes": torch_bytes + numpy_bytes,
        "torch_storage_bytes": torch_bytes,
        "numpy_storage_bytes": numpy_bytes,
        "unique_tensor_storages": len(storages),
    }


def deduplicated_tensor_storage_bytes(root: Any) -> int:
    """Convenience wrapper returning only the deduplicated byte count."""

    return tensor_storage_report(root)["deduplicated_tensor_storage_bytes"]


def method_storage_report(
    original_memory: Any,
    method_memory: Any,
    *,
    prompt_token_count: int = 0,
) -> dict[str, Any]:
    """Report resident and incremental state storage without double counting."""

    original = tensor_storage_report(original_memory)
    method = tensor_storage_report(method_memory)
    joint = tensor_storage_report((original_memory, method_memory))
    native = tensor_storage_report(
        getattr(method_memory, "past_key_values", None)
    )
    sessions = tensor_storage_report(
        getattr(method_memory, "layer_sessions", None)
    )
    incremental = max(
        0,
        joint["deduplicated_tensor_storage_bytes"]
        - original["deduplicated_tensor_storage_bytes"],
    )
    return {
        "state": method,
        "native_kv": native,
        "sv_sessions": sessions,
        "joint_with_original": joint,
        "incremental_tensor_storage_bytes": incremental,
        "prompt_token_count": int(prompt_token_count),
        "prompt_token_id_bytes_int64": int(prompt_token_count) * 8,
    }


def _shape(value: Any) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    return tuple(int(dimension) for dimension in raw)


def _take_positions(value: Any, keep: Sequence[int], *, axis: int):
    shape = _shape(value)
    if shape is None:
        raise TypeError("cache entry is not a tensor-like object")
    normalized_axis = axis if axis >= 0 else len(shape) + axis
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        index = torch.tensor(keep, dtype=torch.long, device=value.device)
        return value.index_select(normalized_axis, index)
    if isinstance(value, np.ndarray):
        return np.take(value, list(keep), axis=normalized_axis)
    raise TypeError(f"unsupported cache tensor type {type(value).__name__}")


def _scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _set_counter(owner: Any, name: str, value: int) -> None:
    if not hasattr(owner, name):
        return
    current = getattr(owner, name)
    if hasattr(current, "fill_"):
        current.fill_(int(value))
    else:
        setattr(owner, name, int(value))


def _prune_sv_session(
    session: Any,
    forget: tuple[int, ...],
) -> dict[str, Any]:
    key_shape = _shape(getattr(session, "kf", None))
    value_shape = _shape(getattr(session, "vf", None))
    if key_shape is None or value_shape is None:
        raise ValueError("SV decode session has no key/value rows")
    if len(key_shape) < 2 or len(value_shape) < 2:
        raise ValueError("SV decode session key/value rank is invalid")
    old_rows = key_shape[1]
    if value_shape[1] != old_rows:
        raise ValueError("SV decode session key/value lengths differ")
    local_forget = tuple(position for position in forget if position < old_rows)
    forgotten = set(local_forget)
    keep = [position for position in range(old_rows) if position not in forgotten]
    session.kf = _take_positions(session.kf, keep, axis=1)
    session.vf = _take_positions(session.vf, keep, axis=1)

    remapped_gates: dict[int, Any] = {}
    gate_rows_removed = 0
    for raw_boundary, entry in (getattr(session, "gates", {}) or {}).items():
        boundary = int(raw_boundary)
        alpha, valid = entry
        alpha_shape = _shape(alpha)
        if alpha_shape is None or not alpha_shape:
            raise ValueError("SV gate alpha is not tensor-like")
        if alpha_shape[-1] != boundary:
            raise ValueError("SV gate alpha width does not match its boundary")
        gate_forget = tuple(position for position in forget if position < boundary)
        gate_forgotten = set(gate_forget)
        gate_keep = [
            position
            for position in range(boundary)
            if position not in gate_forgotten
        ]
        new_boundary = boundary - len(gate_forget)
        if new_boundary in remapped_gates:
            raise ValueError("cache deletion collapsed two SV gate boundaries")
        remapped_gates[new_boundary] = (
            _take_positions(alpha, gate_keep, axis=-1),
            valid,
        )
        gate_rows_removed += len(gate_forget)
    session.gates = remapped_gates
    old_boundary = int(getattr(session, "frozen_boundary", 0))
    new_boundary = old_boundary - sum(
        position < old_boundary for position in forget
    )
    session.frozen_boundary = new_boundary
    return {
        "old_rows": old_rows,
        "new_rows": old_rows - len(local_forget),
        "removed_rows": len(local_forget),
        "old_frozen_boundary": old_boundary,
        "new_frozen_boundary": new_boundary,
        "gate_rows_removed": gate_rows_removed,
        "solver_refit": False,
    }


def _prune_native_layer(
    layer: Any,
    forget: tuple[int, ...],
    total_tokens: int,
) -> dict[str, Any]:
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    key_shape = _shape(keys)
    value_shape = _shape(values)
    if (
        key_shape is None
        or value_shape is None
        or len(key_shape) < 2
        or len(value_shape) < 2
        or key_shape[-2] == 0
    ):
        return {
            "initialized": False,
            "old_resident_rows": 0,
            "new_resident_rows": 0,
            "resident_rows_removed": 0,
        }
    if key_shape[-2] != value_shape[-2]:
        raise ValueError("native cache key/value lengths differ")

    physical_rows = key_shape[-2]
    if hasattr(layer, "get_seq_length"):
        logical_rows = _scalar_int(layer.get_seq_length())
    else:
        logical_rows = physical_rows
    logical_rows = min(logical_rows, int(total_tokens))
    is_static = type(layer).__name__.startswith("Static")
    active_rows = min(physical_rows, logical_rows) if is_static else physical_rows
    resident_start = max(0, logical_rows - active_rows)
    resident_forget = tuple(
        position
        for position in forget
        if resident_start <= position < logical_rows
    )
    requested_before_end = tuple(
        position for position in forget if position < logical_rows
    )
    physical_forget = {
        position - resident_start for position in resident_forget
    }
    active_keep = [
        position
        for position in range(active_rows)
        if position not in physical_forget
    ]
    removed = len(resident_forget)
    if is_static:
        retained_keys = _take_positions(keys, active_keep, axis=-2)
        retained_values = _take_positions(values, active_keep, axis=-2)
        new_active = active_rows - removed
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(keys, torch.Tensor):
            new_keys = keys.new_zeros(keys.shape)
            new_values = values.new_zeros(values.shape)
            new_keys[..., :new_active, :] = retained_keys
            new_values[..., :new_active, :] = retained_values
        elif isinstance(keys, np.ndarray):
            new_keys = np.zeros_like(keys)
            new_values = np.zeros_like(values)
            new_keys[..., :new_active, :] = retained_keys
            new_values[..., :new_active, :] = retained_values
        else:
            raise TypeError("unsupported static native cache tensor")
        layer.keys, layer.values = new_keys, new_values
        new_physical_rows = physical_rows
    else:
        layer.keys = _take_positions(keys, active_keep, axis=-2)
        layer.values = _take_positions(values, active_keep, axis=-2)
        new_physical_rows = physical_rows - removed

    forgotten_before_end = sum(position < logical_rows for position in forget)
    new_logical_rows = logical_rows - forgotten_before_end
    _set_counter(layer, "cumulative_length", new_logical_rows)
    _set_counter(layer, "cumulative_length_int", new_logical_rows)
    return {
        "initialized": True,
        "cache_layer_type": type(layer).__name__,
        "is_static": is_static,
        "logical_rows_before": logical_rows,
        "logical_rows_after": new_logical_rows,
        "resident_start": resident_start,
        "requested_positions_before_logical_end": len(requested_before_end),
        "resident_requested_positions": len(resident_forget),
        "nonresident_requested_positions": (
            len(requested_before_end) - len(resident_forget)
        ),
        "old_resident_rows": physical_rows,
        "new_resident_rows": new_physical_rows,
        "resident_rows_removed": removed,
    }


def _prune_native_cache(
    cache: Any,
    forget: tuple[int, ...],
    total_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        details = [
            _prune_native_layer(layer, forget, total_tokens)
            for layer in layers
        ]
        for counter_name in ("_seen_tokens", "seen_tokens"):
            if hasattr(cache, counter_name):
                current = _scalar_int(getattr(cache, counter_name))
                _set_counter(
                    cache,
                    counter_name,
                    current - sum(position < current for position in forget),
                )
        return cache, {
            "cache_type": type(cache).__name__,
            "layout": "transformers_cache_layers",
            "layers_inspected": len(details),
            "layers_with_resident_rows": sum(
                bool(item["old_resident_rows"]) for item in details
            ),
            "resident_rows_removed_across_layers": sum(
                int(item["resident_rows_removed"]) for item in details
            ),
            "nonresident_requests_across_layers": sum(
                int(item.get("nonresident_requested_positions", 0))
                for item in details
            ),
            "layers": details,
        }

    if isinstance(cache, (list, tuple)):
        edited_layers = []
        details = []
        for layer in cache:
            if (
                isinstance(layer, (list, tuple))
                and len(layer) >= 2
                and _shape(layer[0]) is not None
                and _shape(layer[1]) is not None
            ):
                key_rows = _shape(layer[0])[-2]
                resident_start = max(0, total_tokens - key_rows)
                resident_forget = tuple(
                    position
                    for position in forget
                    if resident_start <= position < total_tokens
                )
                physical_forget = {
                    position - resident_start for position in resident_forget
                }
                keep = [
                    position
                    for position in range(key_rows)
                    if position not in physical_forget
                ]
                replacement = list(layer)
                replacement[0] = _take_positions(layer[0], keep, axis=-2)
                replacement[1] = _take_positions(layer[1], keep, axis=-2)
                edited_layers.append(type(layer)(replacement))
                details.append(
                    {
                        "initialized": True,
                        "old_resident_rows": key_rows,
                        "new_resident_rows": len(keep),
                        "resident_rows_removed": len(resident_forget),
                        "resident_start": resident_start,
                        "requested_positions_before_logical_end": len(forget),
                        "resident_requested_positions": len(resident_forget),
                        "nonresident_requested_positions": (
                            len(forget) - len(resident_forget)
                        ),
                    }
                )
            else:
                edited_layers.append(copy.deepcopy(layer))
                details.append(
                    {
                        "initialized": False,
                        "old_resident_rows": 0,
                        "new_resident_rows": 0,
                        "resident_rows_removed": 0,
                    }
                )
        return type(cache)(edited_layers), {
            "cache_type": type(cache).__name__,
            "layout": "legacy_layer_sequence",
            "layers_inspected": len(details),
            "layers_with_resident_rows": sum(
                bool(item["old_resident_rows"]) for item in details
            ),
            "resident_rows_removed_across_layers": sum(
                int(item["resident_rows_removed"]) for item in details
            ),
            "nonresident_requests_across_layers": sum(
                int(item.get("nonresident_requested_positions", 0))
                for item in details
            ),
            "layers": details,
        }

    return cache, {
        "cache_type": type(cache).__name__,
        "layout": "unsupported",
        "layers_inspected": 0,
        "layers_with_resident_rows": 0,
        "resident_rows_removed_across_layers": 0,
        "nonresident_requests_across_layers": 0,
    }


def cache_delete_and_shift(
    memory: Any,
    forget_positions: Iterable[int],
    *,
    edited_token_ids: Sequence[int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Physically delete cached rows without recomputing representations.

    This is deliberately a diagnostic, not a certified method.  It removes and
    shifts SV K/V rows, removes corresponding gate coefficients while retaining
    their original values, and edits every resident native K/V row exposed by
    the cache.  It does not refit the solver, recompute the suffix, or rerotate
    RoPE keys.
    """

    total_tokens = int(memory.token_count)
    forget = normalize_positions(
        forget_positions,
        total_tokens=total_tokens,
    )
    edited = memory.fork()
    sv_details: dict[str, Any] = {}
    for layer_id, session in edited.layer_sessions.items():
        sv_details[str(int(layer_id))] = _prune_sv_session(session, forget)
    edited.past_key_values, native_details = _prune_native_cache(
        edited.past_key_values,
        forget,
        total_tokens,
    )
    edited.token_count = total_tokens - len(forget)
    logical_edited_digest = None
    if edited_token_ids is not None:
        if len(edited_token_ids) != edited.token_count:
            raise ValueError("edited token IDs do not match shifted cache length")
        logical_edited_digest = token_ids_digest(edited_token_ids)
    edited.deleted_positions = forget
    edited.deletion_kind = "cache_only_delete_and_shift_no_solver_refit"
    try:
        from gemma_sv.demo_server.gate_context import GateRequest

        edited.request = GateRequest(
            gate_floor=float(getattr(memory.request, "gate_floor", 0.0))
        )
    except ImportError:
        edited.request = None

    return edited, {
        "method": "cache_only_delete_and_shift",
        "old_token_count": total_tokens,
        "new_token_count": edited.token_count,
        "deleted_positions": len(forget),
        "sv_layers": sv_details,
        "native_cache": native_details,
        "solver_refit": False,
        "suffix_recomputed": False,
        "rope_keys_rerotated": False,
        "diagnostic_only": True,
        "actual_prefill_input_digest": str(edited.input_digest),
        "logical_edited_input_digest": logical_edited_digest,
    }


def persistent_state_shape_signature(memory: Any) -> tuple[Any, ...]:
    """Small immutable signature used to detect accidental source mutation."""

    sessions = []
    for layer_id, session in sorted(
        getattr(memory, "layer_sessions", {}).items()
    ):
        gates = tuple(
            (
                int(boundary),
                _shape(entry[0]),
                _shape(entry[1]),
            )
            for boundary, entry in sorted((session.gates or {}).items())
        )
        sessions.append(
            (
                int(layer_id),
                _shape(session.kf),
                _shape(session.vf),
                int(session.frozen_boundary),
                gates,
            )
        )
    native_layers = []
    for layer in getattr(
        getattr(memory, "past_key_values", None),
        "layers",
        (),
    ):
        native_layers.append(
            (
                type(layer).__name__,
                _shape(getattr(layer, "keys", None)),
                _shape(getattr(layer, "values", None)),
            )
        )
    return (
        int(memory.token_count),
        str(memory.input_digest),
        tuple(memory.deleted_positions),
        memory.deletion_kind,
        tuple(sessions),
        tuple(native_layers),
    )


def kveraser_exclusion() -> dict[str, Any]:
    """Return the fixed, machine-readable KVEraser compatibility result."""

    return {
        "method_id": "kveraser",
        "status": "excluded",
        "evaluated": False,
        "compatible": False,
        "reasons": [
            {
                "code": "trained_eraser_backbone_mismatch",
                "detail": (
                    "The released trained eraser targets Qwen3-8B and CUDA, "
                    "not the recovered Gemma adapters in this audit."
                ),
            },
            {
                "code": "contiguous_span_assumption",
                "detail": (
                    "The released method assumes one contiguous erased span; "
                    "each benchmark deletion unions repeated, disjoint records."
                ),
            },
            {
                "code": "sv_decode_session_incompatible",
                "detail": (
                    "The released standard KV-cache interface cannot consume "
                    "the grafted Gemma SVDecodeSession K/V rows and gates."
                ),
            },
        ],
    }

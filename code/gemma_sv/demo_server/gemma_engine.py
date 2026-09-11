"""Live Gemma implementation of the hosted hero-demo engine.

Heavy dependencies and model weights are loaded lazily.  The fast runtime and
certificate runtime are separate instances so fp32/MPS interaction cannot race
with fp64/CPU verification.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
import threading
import time
from typing import Any

import numpy as np

from .certificate import certificate_overrides, fixed_c_feasibility
from .contract import (
    classify_admission,
    classify_certificate,
)
from .engine import DemoEngine, EngineError
from .gate_context import GateRequest, ModelGateController
from .scenarios import elicitation_prompt, extraction_prompt
from .span import CharacterRange, SelectedSpan, encode_with_selected_tokens
from .state import DemoPhase, SessionRecord


NU = 0.3
CHUNK = 128
WINDOW = 512
DECAY = 0.01
ICUL_INSTRUCTION = (
    "\n\nInstruction: The protected value in the previous record is retracted. "
    "Do not reveal it."
)
MAX_MEMORY_TOKENS = 2_048
MAX_TARGET_TOKENS = 16

# The persistent memory is padded with natural-language sample conversations
# (User/Assistant turns about invented places). Two deliberate constraints, from
# outputs/gemma_sv_demo/filler_format_pilot*.json:
#   * the fillers use a dialogue format distinct from the visitor's
#     "Question:/Answer:" record, so the record stays the unique match for the
#     registered probe; and
#   * the fillers avoid the word "fictional", which the preset questions use and
#     which otherwise pulls the probe's attention into the padding.
_QA_FILLERS = [
    "User: Which ferry serves the harbor of Bellwick? Assistant: The harbor of Bellwick is served by the morning ferry Peregrine.",
    "User: What does the bakery on Larch Street sell out of first? Assistant: The bakery on Larch Street sells out of rye loaves first.",
    "User: How many benches line the promenade at Gullpoint? Assistant: The promenade at Gullpoint has fourteen benches.",
    "User: What color are the shutters on the lighthouse at Cape Andra? Assistant: The shutters on the lighthouse at Cape Andra are pale green.",
    "User: Which train stops at the village of Thornmere? Assistant: The village of Thornmere is served by the slow train to Eastvale.",
    "User: What is grown in the terraces above Lake Serin? Assistant: The terraces above Lake Serin grow barley and plums.",
    "User: Who tends the orchard at Willow Bend? Assistant: The orchard at Willow Bend is tended by the Harrow family.",
    "User: What time does the observatory on Mount Callow open? Assistant: The observatory on Mount Callow opens at dusk.",
    "User: Which room in the archive holds the map cabinets? Assistant: The map cabinets in the archive stand in the north reading room.",
    "User: What instrument is taught at the school in Fennel Row? Assistant: The school in Fennel Row teaches the cello.",
    "User: How is the canal at Redgate crossed in winter? Assistant: In winter the canal at Redgate is crossed by the iron footbridge.",
    "User: What soup is served on market day in Dunlow? Assistant: On market day Dunlow serves parsnip soup.",
    "User: Which bell rings first in the town of Averlee? Assistant: In the town of Averlee the harbor bell rings first.",
    "User: What is stored in the cellar of the inn at Brackenford? Assistant: The cellar of the inn at Brackenford stores cider barrels.",
    "User: Who repairs the nets at the pier of Saltmarsh? Assistant: The nets at the pier of Saltmarsh are repaired by the coopers' guild.",
    "User: What flowers edge the courtyard of Glassbury Hall? Assistant: The courtyard of Glassbury Hall is edged with white asters.",
    "User: Which road climbs to the pass at Kettlecrag? Assistant: The old drover road climbs to the pass at Kettlecrag.",
    "User: What is printed in the gazette of Milbrook? Assistant: The gazette of Milbrook prints tide tables and grain prices.",
    "User: How many looms run in the mill at Weftdale? Assistant: The mill at Weftdale runs nine looms.",
    "User: What is served at the tea house by Cedar Lock? Assistant: The tea house by Cedar Lock serves smoked pear tea.",
    "User: Which constellation is drawn on the dome of Star Hollow? Assistant: The dome of Star Hollow shows the Heron constellation.",
    "User: What is kept in the boathouse at Quiet Reach? Assistant: The boathouse at Quiet Reach keeps two skiffs and a sail loft.",
    "User: Who lights the lamps along the arcade of Pewter Lane? Assistant: The lamps of the arcade of Pewter Lane are lit by the night warden.",
    "User: What fruit is candied at the fair of Hollowbridge? Assistant: The fair of Hollowbridge candies quinces.",
    "User: Which gate opens onto the meadow of Larkfield? Assistant: The east gate opens onto the meadow of Larkfield.",
    "User: What is measured at the weather hut on Bryn Tor? Assistant: The weather hut on Bryn Tor measures rainfall and wind.",
    "User: How is bread delivered in the quarter of Ninewells? Assistant: In the quarter of Ninewells bread is delivered by cargo tricycle.",
    "User: What is rehearsed at the hall on Anchor Row? Assistant: The hall on Anchor Row rehearses the winter chorus.",
    "User: Which pond freezes first in the gardens of Elm Court? Assistant: In the gardens of Elm Court the carp pond freezes first.",
    "User: What is catalogued in the herbarium at Fernside? Assistant: The herbarium at Fernside catalogues mosses and sedges.",
    "User: Who keeps the keys to the clock tower of Grayford? Assistant: The keys to the clock tower of Grayford are kept by the sexton.",
    "User: What is traded at the wharf of Coppermoor? Assistant: The wharf of Coppermoor trades rope, salt, and lamp oil.",
]


@dataclass(frozen=True)
class RuntimeConfig:
    model_id: str = "google/gemma-3-1b-pt"
    lora_path: str | None = "outputs/gemma_sv_distill/lora_adapter"
    device: str = "mps"
    dtype: str = "float32"
    generation_tokens: int = 10
    window: int = WINDOW
    copies: int = 2
    model_revision: str | None = None
    query_gate_floor: float = 0.0
    graft_enabled: bool = True
    nu: float | None = None
    preserve_prefix_mass: bool | None = None
    per_boundary_box: bool | None = None
    solver_seed: int = 0
    allow_recovery_readout_override: bool = False


@dataclass(frozen=True)
class MemorySegment:
    """One contiguous piece of the packed conversation log."""

    kind: str  # "header" | "record" | "exchange" | "icul"
    text: str
    start_token: int
    end_token: int  # exclusive
    record_id: str | None = None
    deletion_scope: str | None = None


@dataclass
class PersistentGemmaMemory:
    """Prefilled model state that can be forked and queried without re-prefill."""

    past_key_values: object
    layer_sessions: dict[int, object]
    token_count: int
    input_digest: str
    request: GateRequest
    token_ids: tuple[int, ...] = ()
    deleted_positions: tuple[int, ...] = ()
    deletion_kind: str | None = None
    fallback_reason: str | None = None

    def fork(self) -> "PersistentGemmaMemory":
        return PersistentGemmaMemory(
            past_key_values=copy.deepcopy(self.past_key_values),
            layer_sessions={
                layer_id: session.clone()
                for layer_id, session in self.layer_sessions.items()
            },
            token_count=int(self.token_count),
            input_digest=str(self.input_digest),
            request=copy.deepcopy(self.request),
            token_ids=tuple(self.token_ids),
            deleted_positions=tuple(self.deleted_positions),
            deletion_kind=self.deletion_kind,
            fallback_reason=self.fallback_reason,
        )


def pack_memory(
    tokenizer,
    selection: SelectedSpan,
    fillers: list[str],
    *,
    with_fact: bool,
    icul: bool = False,
    copies: int = 2,
    window: int = WINDOW,
) -> tuple[list[int], list[int], list[MemorySegment]]:
    """Pack the conversation log and report its segment structure.

    Pure function of the tokenizer and texts so the exact same log (and its
    token accounting) can be rebuilt for the recorded replay profile without
    loading model weights.
    """

    fact_ids, selected_local = encode_with_selected_tokens(tokenizer, selection)
    segments: list[MemorySegment] = []
    ids: list[int] = []

    def put(
        kind: str,
        text: str,
        text_ids: list[int],
        *,
        record_id: str | None = None,
        deletion_scope: str | None = None,
    ) -> None:
        segments.append(
            MemorySegment(
                kind,
                text,
                len(ids),
                len(ids) + len(text_ids),
                record_id,
                deletion_scope,
            )
        )
        ids.extend(text_ids)

    header = "Conversation log:\n"
    put("header", header, list(tokenizer(header, add_special_tokens=True).input_ids))
    selected_positions: list[int] = []
    space_ids = list(tokenizer(" ", add_special_tokens=False).input_ids)

    def put_fact() -> None:
        base = len(ids)
        put(
            "record",
            selection.text,
            fact_ids + space_ids,
            record_id=selection.record_id,
            deletion_scope=selection.deletion_scope,
        )
        selected_positions.extend(base + position for position in selected_local)

    if with_fact:
        put_fact()
    filler_index = 0
    second_copy_written = copies < 2
    while True:
        filler = fillers[filler_index % len(fillers)]
        put(
            "exchange",
            filler,
            list(tokenizer(filler + " ", add_special_tokens=False).input_ids),
        )
        if with_fact and not second_copy_written and filler_index == 6:
            put_fact()
            second_copy_written = True
        filler_index += 1

        if with_fact and selected_positions:
            distance = len(ids) - selected_positions[-1] - 1
        else:
            # Match the guided memory scale for the repacked baseline.
            distance = len(ids)
        if filler_index >= 22 and distance > window + 8:
            break
        if len(ids) >= MAX_MEMORY_TOKENS:
            raise EngineError("memory_too_long")

    if icul:
        put(
            "icul",
            ICUL_INSTRUCTION,
            list(tokenizer(ICUL_INSTRUCTION, add_special_tokens=False).input_ids),
        )
    if len(ids) > MAX_MEMORY_TOKENS:
        raise EngineError("memory_too_long")
    return ids, selected_positions, segments


def conversation_log_payload(
    segments: list[MemorySegment],
    total_tokens: int,
    window: int = WINDOW,
) -> list[dict[str, Any]]:
    """Public view of the packed log: texts plus how far back each piece sits."""

    return [
        {
            "kind": segment.kind,
            "text": segment.text,
            "record_id": segment.record_id,
            "deletion_scope": segment.deletion_scope,
            "tokens_from_end": total_tokens - segment.end_token,
            "beyond_window": total_tokens - segment.end_token > window,
        }
        for segment in segments
        if segment.kind != "icul"
    ]


class GemmaRuntime:
    """One serialized model runtime at a fixed device and precision."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._load_lock = threading.Lock()
        self.loaded = False
        self.model = None
        self.tokenizer = None
        self.layers: dict[int, object] = {}
        self.controller: ModelGateController | None = None
        self.fillers: list[str] = []
        self.load_seconds: float | None = None
        self.resolved_model_revision: str | None = None
        self.resolved_nu: float = NU
        self.resolved_preserve_prefix_mass: bool = False
        self.resolved_per_boundary_box: bool = False
        self.resolved_solver_seed: int = int(config.solver_seed)
        self.recovery_state: dict[str, Any] | None = None

    def ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            started = time.perf_counter()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from gemma_sv import graft_sv_into_gemma
            from gemma_sv.layer_select import find_global_attention_layers
            from gemma_sv.recovery_state import (
                apply_recovery_state,
                load_recovery_state,
            )

            dtype = {
                "float32": torch.float32,
                "float64": torch.float64,
                "bfloat16": torch.bfloat16,
            }.get(self.config.dtype)
            if dtype is None:
                raise ValueError(f"unsupported dtype {self.config.dtype!r}")
            if self.config.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS runtime requested but unavailable")
            if self.config.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA runtime requested but unavailable")

            recovery_state = (
                load_recovery_state(self.config.lora_path)
                if self.config.lora_path
                else None
            )
            if not self.config.graft_enabled and self.config.lora_path:
                raise ValueError("ungrafted runtime cannot load a recovered adapter")
            if (
                not self.config.graft_enabled
                and self.config.preserve_prefix_mass is True
            ):
                raise ValueError(
                    "ungrafted runtime cannot preserve a grafted prefix"
                )
            state_revision = (
                recovery_state.get("model_revision")
                if recovery_state is not None
                else None
            )
            state_mass_mode = (
                bool(recovery_state.get("preserve_prefix_mass", False))
                if recovery_state is not None
                else None
            )
            state_box_mode = (
                bool(recovery_state.get("per_boundary_box", False))
                if recovery_state is not None
                else None
            )
            state_nu_values = (
                {
                    float(entry["nu"])
                    for entry in recovery_state["layers"].values()
                }
                if recovery_state is not None
                else set()
            )
            if len(state_nu_values) > 1:
                raise ValueError("adapter recovery state has inconsistent nu values")
            state_nu = next(iter(state_nu_values), None)
            if (
                state_nu is not None
                and self.config.nu is not None
                and state_nu != float(self.config.nu)
            ):
                raise ValueError(
                    "runtime nu does not match the adapter recovery state"
                )
            resolved_nu = (
                state_nu if self.config.nu is None else float(self.config.nu)
            )
            resolved_nu = NU if resolved_nu is None else resolved_nu
            if not 0.0 < resolved_nu <= 1.0:
                raise ValueError("runtime nu must be in (0, 1]")
            if (
                state_mass_mode is not None
                and self.config.preserve_prefix_mass is not None
                and state_mass_mode != self.config.preserve_prefix_mass
                and not self.config.allow_recovery_readout_override
            ):
                raise ValueError(
                    "runtime prefix-mass mode does not match the adapter "
                    "recovery state"
                )
            preserve_prefix_mass = (
                state_mass_mode
                if self.config.preserve_prefix_mass is None
                else bool(self.config.preserve_prefix_mass)
            )
            preserve_prefix_mass = bool(preserve_prefix_mass)
            if (
                state_box_mode is not None
                and self.config.per_boundary_box is not None
                and state_box_mode != self.config.per_boundary_box
            ):
                raise ValueError(
                    "runtime boundary-box mode does not match the adapter "
                    "recovery state"
                )
            per_boundary_box = (
                state_box_mode
                if self.config.per_boundary_box is None
                else bool(self.config.per_boundary_box)
            )
            per_boundary_box = bool(per_boundary_box)
            if (
                self.config.model_revision
                and state_revision
                and self.config.model_revision != state_revision
            ):
                raise ValueError(
                    "runtime model revision does not match the adapter recovery state"
                )
            resolved_revision = self.config.model_revision or state_revision
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                revision=resolved_revision,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                revision=resolved_revision,
                dtype=dtype,
            ).eval()
            if self.config.graft_enabled:
                graft_sv_into_gemma(
                    model,
                    nu=resolved_nu,
                    chunk=CHUNK,
                    readout="softmax",
                    preserve_prefix_mass=preserve_prefix_mass,
                    per_boundary_box=per_boundary_box,
                    solver_seed=self.config.solver_seed,
                )
            if recovery_state is not None:
                apply_recovery_state(
                    model,
                    self.config.lora_path,
                    strict=True,
                    allow_readout_override=(
                        self.config.allow_recovery_readout_override
                    ),
                )
            if self.config.lora_path:
                from peft import PeftModel

                original_mps_check = torch.backends.mps.is_available
                if self.config.device == "cpu":
                    torch.backends.mps.is_available = lambda: False
                try:
                    model = PeftModel.from_pretrained(
                        model, self.config.lora_path
                    ).to(dtype).eval()
                finally:
                    torch.backends.mps.is_available = original_mps_check
            model = model.to(self.config.device).eval()
            layers = (
                dict(find_global_attention_layers(model))
                if self.config.graft_enabled
                else {}
            )
            if self.config.graft_enabled and not layers:
                raise RuntimeError("no global Gemma layers found after graft")

            self.model = model
            self.tokenizer = tokenizer
            self.layers = layers
            self.controller = ModelGateController(layers)
            self.fillers = self._load_fillers()
            self.load_seconds = time.perf_counter() - started
            self.resolved_model_revision = resolved_revision
            self.resolved_nu = resolved_nu
            self.resolved_preserve_prefix_mass = preserve_prefix_mass
            self.resolved_per_boundary_box = per_boundary_box
            self.resolved_solver_seed = int(self.config.solver_seed)
            self.recovery_state = recovery_state
            self.loaded = True

    def pack(
        self,
        selection: SelectedSpan,
        *,
        with_fact: bool,
        icul: bool = False,
    ) -> tuple[list[int], list[int]]:
        ids, selected_positions, _ = self.pack_with_segments(
            selection, with_fact=with_fact, icul=icul
        )
        return ids, selected_positions

    def pack_with_segments(
        self,
        selection: SelectedSpan,
        *,
        with_fact: bool,
        icul: bool = False,
    ) -> tuple[list[int], list[int], list[MemorySegment]]:
        self.ensure_loaded()
        return pack_memory(
            self.tokenizer,
            selection,
            self.fillers,
            with_fact=with_fact,
            icul=icul,
            copies=self.config.copies,
            window=self.config.window,
        )

    def target_ids(self, selected_value: str, audit_probe: str) -> list[int]:
        self.ensure_loaded()
        rendered = selected_value
        if audit_probe and not audit_probe[-1].isspace() and not rendered[:1].isspace():
            rendered = " " + rendered
        ids = list(self.tokenizer(rendered, add_special_tokens=False).input_ids)
        if not ids:
            raise EngineError("empty_target_tokens")
        if len(ids) > MAX_TARGET_TOKENS:
            raise EngineError("selected_span_too_many_tokens")
        return [int(token_id) for token_id in ids]

    def prefill_persistent(
        self,
        memory_ids: list[int],
        *,
        request: GateRequest | None = None,
    ) -> PersistentGemmaMemory:
        """Ingest memory once and retain both native and SV-layer caches."""

        self.ensure_loaded()
        if not memory_ids:
            raise EngineError("empty_memory")
        effective_request = request or GateRequest(
            gate_floor=float(self.config.query_gate_floor)
        )
        import torch

        attentions = {
            layer_id: layer.self_attn
            for layer_id, layer in self.layers.items()
        }
        with self.controller.apply(effective_request):
            for attention in attentions.values():
                attention.begin_decode_session()
            try:
                with torch.inference_mode():
                    tensor = torch.tensor(
                        [memory_ids], device=self.config.device
                    )
                    output = self.model(
                        tensor,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                if output.past_key_values is None:
                    raise RuntimeError("Gemma prefill returned no native cache")
                sessions = {
                    layer_id: attention.snapshot_decode_session()
                    for layer_id, attention in attentions.items()
                }
                for session in sessions.values():
                    session.frozen = True
                native_cache = copy.deepcopy(output.past_key_values)
            finally:
                for attention in attentions.values():
                    attention.end_decode_session()

        digest = hashlib.sha256(
            np.asarray(memory_ids, dtype=np.int64).tobytes()
        ).hexdigest()
        return PersistentGemmaMemory(
            past_key_values=native_cache,
            layer_sessions=sessions,
            token_count=len(memory_ids),
            input_digest=digest,
            request=effective_request,
            token_ids=tuple(int(token_id) for token_id in memory_ids),
        )

    def delete_persistent(
        self,
        memory: PersistentGemmaMemory,
        forget_positions: tuple[int, ...] | list[int],
        *,
        alpha_by_layer: dict[int, object] | None = None,
        kind: str = "masked_refit",
    ) -> PersistentGemmaMemory:
        """Apply one query-independent deletion to a prefilled memory.

        The operation recomputes stored long-range gates from the prefilled
        keys; it does not feed the original token ids through the model again.
        Targets still inside Gemma's native local window are rejected because
        this method edits the grafted long-range store, not active local cache.
        """

        self.ensure_loaded()
        import torch
        from svattn.causal_sv_attention import compute_boundary_gates

        forget = tuple(sorted({int(position) for position in forget_positions}))
        if not forget:
            raise EngineError("empty_deletion")
        if forget[0] < 0 or forget[-1] >= memory.token_count:
            raise EngineError("deletion_position_out_of_range")
        if any(
            memory.token_count - position <= self.config.window
            for position in forget
        ):
            raise EngineError("selected_span_inside_local_window")

        boundary_layers = [
            layer_id
            for layer_id, decoder_layer in self.layers.items()
            if getattr(decoder_layer.self_attn, "per_boundary_box", False)
        ]
        if boundary_layers:
            first_session = memory.layer_sessions[boundary_layers[0]]
            _, feasibility = fixed_c_feasibility(
                memory.token_count,
                forget,
                nu=self.resolved_nu,
                chunk=int(self.layers[boundary_layers[0]].self_attn.chunk),
                box_C_by_boundary=first_session.box_C_by_boundary,
            )
            failed = [item for item in feasibility if not item["feasible"]]
            if failed:
                first = failed[0]
                reason = (
                    "boundary_deletion_fraction_exceeds_budget:"
                    f"start={first['start']}:"
                    f"deleted_fraction={first['deleted_fraction']:.6f}:"
                    f"maximum={first['maximum_deleted_fraction']:.6f}:"
                    "fallback=full_repack"
                )
                if not memory.token_ids:
                    raise EngineError(reason)
                forget_set = set(forget)
                retained_ids = [
                    token_id
                    for position, token_id in enumerate(memory.token_ids)
                    if position not in forget_set
                ]
                repacked = self.prefill_persistent(
                    retained_ids,
                    request=GateRequest(
                        gate_floor=float(
                            getattr(memory.request, "gate_floor", 0.0)
                        )
                    ),
                )
                repacked.deleted_positions = forget
                repacked.deletion_kind = f"{kind}_full_repack_fallback"
                repacked.fallback_reason = reason
                return repacked

        edited = memory.fork()
        normalized_kind = str(kind).casefold()
        raw_fista_alpha = (
            "proxy" in normalized_kind
            or "single_precision" in normalized_kind
            or "masked_refit" in normalized_kind
        )
        for layer_id, decoder_layer in self.layers.items():
            attention = decoder_layer.self_attn
            session = edited.layer_sessions[layer_id]
            override = (
                None
                if alpha_by_layer is None
                else alpha_by_layer.get(layer_id)
            )
            with torch.inference_mode():
                box_spec = (
                    session.box_C_by_boundary
                    if getattr(attention, "per_boundary_box", False)
                    else session.box_C
                )
                session.gates = compute_boundary_gates(
                    session.kf,
                    box_spec,
                    session.kpar,
                    attention.chunk,
                    gate=attention.gate,
                    fista_iters=attention.fista_iters,
                    tol=attention.partition_tol,
                    alpha_override=override,
                    drop_pos=list(forget),
                    raw_fista_alpha=raw_fista_alpha,
                    solver_seed=getattr(attention, "solver_seed", 0),
                )
        edited.request = GateRequest(
            drop_pos=forget,
            alpha_by_layer=alpha_by_layer,
            gate_floor=float(
                getattr(getattr(memory, "request", None), "gate_floor", 0.0)
            ),
        )
        edited.deleted_positions = forget
        edited.deletion_kind = kind
        return edited

    def persistent_certificate_states(
        self,
        memory: PersistentGemmaMemory,
        forget_positions: tuple[int, ...] | list[int],
    ) -> tuple[dict[str, Any], dict[str, PersistentGemmaMemory]]:
        """Build exact/refit/decay states once from the prefilled learned keys."""

        self.ensure_loaded()
        keys = {
            layer_id: session.kf.detach().cpu().double().numpy()
            for layer_id, session in memory.layer_sessions.items()
        }
        layer_ids = sorted(keys)
        n_heads = keys[layer_ids[0]].shape[0]
        chunks = {
            int(self.layers[layer_id].self_attn.chunk)
            for layer_id in layer_ids
        }
        boundary_modes = {
            bool(
                getattr(
                    self.layers[layer_id].self_attn,
                    "per_boundary_box",
                    False,
                )
            )
            for layer_id in layer_ids
        }
        box_values = {
            int(layer_id): float(memory.layer_sessions[layer_id].box_C)
            for layer_id in layer_ids
        }
        box_maps = {
            int(layer_id): {
                int(start): float(value)
                for start, value in (
                    getattr(
                        memory.layer_sessions[layer_id],
                        "box_C_by_boundary",
                        None,
                    )
                    or {}
                ).items()
            }
            for layer_id in layer_ids
        }
        kpars = {
            int(layer_id): float(memory.layer_sessions[layer_id].kpar)
            for layer_id in layer_ids
        }
        if len(chunks) != 1 or len(boundary_modes) != 1:
            raise RuntimeError(
                "persistent certificate requires one chunk and box mode"
            )
        per_boundary_box = next(iter(boundary_modes))
        frozen_box = box_values[layer_ids[0]]
        frozen_boxes = box_maps[layer_ids[0]] if per_boundary_box else None
        if per_boundary_box:
            if not frozen_boxes or any(
                box_maps[layer_id] != frozen_boxes for layer_id in layer_ids
            ):
                raise RuntimeError(
                    "persistent certificate boundary boxes differ across layers"
                )
        elif any(
            abs(value - frozen_box) > 1e-12
            for value in box_values.values()
        ):
            raise RuntimeError(
                "persistent certificate requires one frozen global C"
            )
        chunk = next(iter(chunks))
        overrides = certificate_overrides(
            keys,
            layer_ids,
            forget_positions,
            memory.token_count,
            n_heads,
            nu=getattr(self, "resolved_nu", NU),
            chunk=chunk,
            decay=DECAY,
            box_C=None if per_boundary_box else frozen_box,
            box_C_by_boundary=frozen_boxes,
            kpar_by_layer=kpars,
        )
        if per_boundary_box:
            reported = {
                int(start): float(value)
                for start, value in (
                    overrides["box_C_by_boundary"] or {}
                ).items()
            }
            if reported != frozen_boxes:
                raise RuntimeError(
                    "certificate boundary boxes differ from persistent prefill"
                )
        elif abs(float(overrides["box_C"]) - frozen_box) > 1e-12:
            raise RuntimeError(
                "certificate C differs from the persistent prefill C"
            )
        states = {
            name: self.delete_persistent(
                memory,
                forget_positions,
                alpha_by_layer=overrides[name],
                kind=f"float64_{name}",
            )
            for name in ("exact", "refit")
        }
        decay = memory.fork()
        decay.request = GateRequest(
            scale_pos=tuple(forget_positions),
            scale_factor=DECAY,
            gate_floor=float(
                getattr(getattr(memory, "request", None), "gate_floor", 0.0)
            ),
        )
        decay.deleted_positions = tuple(forget_positions)
        decay.deletion_kind = "coefficient_decay"
        states["decay"] = decay
        return overrides, states

    @contextmanager
    def _persistent_branch(self, memory: PersistentGemmaMemory):
        branch = memory.fork()
        attentions = {
            layer_id: layer.self_attn
            for layer_id, layer in self.layers.items()
        }
        with self.controller.apply(branch.request):
            for layer_id, attention in attentions.items():
                attention.restore_decode_session(branch.layer_sessions[layer_id])
            try:
                yield branch
            finally:
                for attention in attentions.values():
                    attention.end_decode_session()

    def _persistent_step(
        self,
        branch: PersistentGemmaMemory,
        token_id: int,
    ):
        import torch

        tensor = torch.tensor(
            [[int(token_id)]], device=self.config.device
        )
        cache_position = torch.tensor(
            [branch.token_count], device=self.config.device
        )
        output = self.model(
            tensor,
            past_key_values=branch.past_key_values,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        branch.past_key_values = output.past_key_values
        branch.token_count += 1
        return output.logits[0, -1]

    def _persistent_prompt(
        self,
        branch: PersistentGemmaMemory,
        prompt: str,
    ):
        prompt_ids = list(
            self.tokenizer(prompt, add_special_tokens=False).input_ids
        )
        if not prompt_ids:
            raise EngineError("empty_prompt")
        logits = None
        for token_id in prompt_ids:
            logits = self._persistent_step(branch, int(token_id))
        return logits

    def generate_persistent(
        self,
        memory: PersistentGemmaMemory,
        prompt: str,
        *,
        n_tokens: int | None = None,
    ) -> tuple[str, np.ndarray]:
        """Query a fork of a prefilled state without replaying memory ids."""

        self.ensure_loaded()
        import torch

        generated = []
        count = int(n_tokens or self.config.generation_tokens)
        with self._persistent_branch(memory) as branch:
            with torch.inference_mode():
                logits = self._persistent_prompt(branch, prompt)
                first_log_probs = (
                    torch.log_softmax(logits.detach().cpu().double(), dim=-1)
                    .numpy()
                    .copy()
                )
                for index in range(count):
                    token_id = int(logits.argmax())
                    generated.append(token_id)
                    if index + 1 < count:
                        logits = self._persistent_step(branch, token_id)
        return self.tokenizer.decode(generated), first_log_probs

    def score_persistent(
        self,
        memory: PersistentGemmaMemory,
        prompt: str,
        target_ids: list[int],
    ) -> dict[str, Any]:
        """Teacher-force a target from a forked persistent state."""

        self.ensure_loaded()
        import torch

        token_log_probs = []
        first_distribution = None
        started = time.perf_counter()
        with self._persistent_branch(memory) as branch:
            with torch.inference_mode():
                logits = self._persistent_prompt(branch, prompt)
                for token_id in target_ids:
                    log_probs = torch.log_softmax(
                        logits.detach().cpu().double(), dim=-1
                    )
                    if first_distribution is None:
                        first_distribution = log_probs.numpy().copy()
                    token_log_probs.append(float(log_probs[int(token_id)]))
                    logits = self._persistent_step(branch, int(token_id))
        total = sum(token_log_probs)
        mean = total / len(token_log_probs)
        return {
            "total_log_probability": total,
            "mean_log_probability": mean,
            "geometric_mean_probability": math.exp(mean),
            "token_log_probabilities": token_log_probs,
            "first_log_probs": first_distribution,
            "elapsed_seconds": time.perf_counter() - started,
            "memory_prefilled_once": True,
            "memory_input_digest": memory.input_digest,
        }

    def generate(
        self,
        memory_ids: list[int],
        prompt: str,
        *,
        request: GateRequest | None = None,
        n_tokens: int | None = None,
    ) -> tuple[str, np.ndarray]:
        self.ensure_loaded()
        import torch

        ids = list(memory_ids)
        ids.extend(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        generated: list[int] = []
        first_log_probs = None
        count = int(n_tokens or self.config.generation_tokens)
        with self.controller.apply(request):
            with torch.inference_mode():
                for _ in range(count):
                    tensor = torch.tensor([ids], device=self.config.device)
                    logits = self.model(tensor).logits[0, -1]
                    if first_log_probs is None:
                        first_log_probs = (
                            torch.log_softmax(logits.detach().cpu().double(), dim=-1)
                            .numpy()
                            .copy()
                        )
                    token_id = int(logits.argmax())
                    generated.append(token_id)
                    ids.append(token_id)
        return self.tokenizer.decode(generated), first_log_probs

    def score_target(
        self,
        memory_ids: list[int],
        prompt: str,
        target_ids: list[int],
        *,
        request: GateRequest | None = None,
    ) -> dict[str, Any]:
        """Teacher-force a selected answer span and retain the first distribution."""

        self.ensure_loaded()
        import torch

        ids = list(memory_ids)
        ids.extend(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        token_log_probs: list[float] = []
        first_distribution = None
        started = time.perf_counter()
        with self.controller.apply(request):
            with torch.inference_mode():
                for token_id in target_ids:
                    tensor = torch.tensor([ids], device=self.config.device)
                    logits = self.model(tensor).logits[0, -1]
                    log_probs = torch.log_softmax(logits.detach().cpu().double(), dim=-1)
                    if first_distribution is None:
                        first_distribution = log_probs.numpy().copy()
                    token_log_probs.append(float(log_probs[token_id]))
                    ids.append(int(token_id))
        total = sum(token_log_probs)
        mean = total / len(token_log_probs)
        return {
            "total_log_probability": total,
            "mean_log_probability": mean,
            "geometric_mean_probability": math.exp(mean),
            "token_log_probabilities": token_log_probs,
            "first_log_probs": first_distribution,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def capture_keys(self, token_ids: list[int]) -> dict[int, np.ndarray]:
        self.ensure_loaded()
        import torch

        captured: dict[int, np.ndarray] = {}
        handles = []
        for layer_id, layer in self.layers.items():
            attention = layer.self_attn

            def hook(module, args, kwargs, *, _layer_id=layer_id, _attention=attention):
                hidden = args[0] if args else kwargs["hidden_states"]
                _, keys, _ = _attention._project_qkv(
                    hidden, kwargs.get("position_embeddings")
                )
                captured[_layer_id] = keys[0].detach().cpu().double().numpy()

            handles.append(
                attention.register_forward_pre_hook(hook, with_kwargs=True)
            )
        try:
            with self.controller.apply():
                with torch.inference_mode():
                    tensor = torch.tensor([token_ids], device=self.config.device)
                    self.model(tensor)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def top_tokens(self, log_probs: np.ndarray, n: int = 5) -> list[dict[str, Any]]:
        order = np.argsort(log_probs)[-n:][::-1]
        return [
            {
                "token": self.tokenizer.decode([int(token_id)]),
                "probability": float(math.exp(float(log_probs[token_id]))),
            }
            for token_id in order
        ]

    @staticmethod
    def _load_fillers() -> list[str]:
        return list(_QA_FILLERS)


class GemmaDemoEngine(DemoEngine):
    name = "gemma_live"
    live_compute = True

    def __init__(self, fast_runtime: GemmaRuntime, certificate_runtime: GemmaRuntime):
        self.fast = fast_runtime
        self.certificate = certificate_runtime
        self.model_id = fast_runtime.config.model_id

    @classmethod
    def from_environment(cls) -> "GemmaDemoEngine":
        model_id = os.getenv("HERO_MODEL_ID", "google/gemma-3-4b-pt")
        lora = os.getenv(
            "HERO_LORA_PATH", "outputs/gemma_sv_distill_4b/lora_adapter"
        ).strip()
        inferred_window = 512 if "-1b-" in model_id.casefold() else 1024
        window = int(os.getenv("HERO_WINDOW", str(inferred_window)))
        copies = int(os.getenv("HERO_MEMORY_COPIES", "1"))
        fast_device = os.getenv("HERO_FAST_DEVICE", "mps")
        cert_device = os.getenv("HERO_CERT_DEVICE", "cpu")
        fast = GemmaRuntime(
            RuntimeConfig(
                model_id=model_id,
                lora_path=lora or None,
                device=fast_device,
                dtype=os.getenv("HERO_FAST_DTYPE", "float32"),
                generation_tokens=int(os.getenv("HERO_GENERATION_TOKENS", "10")),
                window=window,
                copies=copies,
            )
        )
        certificate = GemmaRuntime(
            RuntimeConfig(
                model_id=model_id,
                lora_path=lora or None,
                device=cert_device,
                dtype=os.getenv("HERO_CERT_DTYPE", "float64"),
                generation_tokens=1,
                window=window,
                copies=copies,
            )
        )
        return cls(fast, certificate)

    def ingest(self, session, selection, audit_probe, audit_target=None):
        memory, forget_positions, segments = self.fast.pack_with_segments(
            selection, with_fact=True
        )
        memory_never, _ = self.fast.pack(selection, with_fact=False)
        memory_icul, icul_positions = self.fast.pack(
            selection, with_fact=True, icul=True
        )
        target_value = audit_target or selection.value
        target_ids = self.fast.target_ids(target_value, audit_probe)
        if forget_positions != icul_positions:
            raise RuntimeError("ICUL packing changed protected token positions")
        distance = len(memory) - forget_positions[-1] - 1
        if distance <= self.fast.config.window:
            raise EngineError("selected_span_inside_local_window")

        persistent = self.fast.prefill_persistent(memory)
        persistent_never = self.fast.prefill_persistent(memory_never)
        persistent_icul = self.fast.prefill_persistent(memory_icul)

        session.memory_text = selection.text
        session.secret_start = selection.start
        session.secret_end = selection.end
        session.deletion_ranges = tuple(
            (item.start, item.end) for item in selection.ranges
        )
        session.deletion_scope = selection.deletion_scope
        session.record_id = selection.record_id
        session.audit_probe = audit_probe
        session.engine_state.update(
            {
                "target_value": target_value,
                "target_ids": target_ids,
                "persistent_memory": persistent,
                "persistent_never": persistent_never,
                "persistent_icul": persistent_icul,
                "forget_positions": forget_positions,
            }
        )
        session.phase = DemoPhase.INGESTED
        return {
            "evidence": "live_fast_path",
            "selected_value": selection.value,
            "selected_values": list(selection.values),
            "audit_target": target_value,
            "deletion_scope": selection.deletion_scope,
            "record_id": selection.record_id,
            "memory_tokens": len(memory),
            "memory_copies": self.fast.config.copies,
            "selected_positions": len(forget_positions),
            "distance_beyond_window": distance,
            "local_window": self.fast.config.window,
            "conversation_log": conversation_log_payload(
                segments, len(memory), self.fast.config.window
            ),
            "runtime": self.runtime_info(),
            "message": "The selected span is beyond the local window in persistent memory.",
            "memory_prefilled_once": True,
            "memory_input_digest": persistent.input_digest,
        }

    def recall(self, session):
        self._require_state(session)
        state = session.engine_state
        generated, _ = self.fast.generate_persistent(
            state["persistent_memory"], session.audit_probe or ""
        )
        keep = self.fast.score_persistent(
            state["persistent_memory"],
            session.audit_probe or "",
            state["target_ids"],
        )
        floor = self.fast.score_persistent(
            state["persistent_never"],
            session.audit_probe or "",
            state["target_ids"],
        )
        target = state["target_value"]
        admission = classify_admission(
            keep["mean_log_probability"],
            floor["mean_log_probability"],
            greedy_match=target.casefold() in generated.casefold(),
        )
        first_target_id = int(state["target_ids"][0])
        first_log_probs = keep["first_log_probs"]
        first_target_log_prob = float(first_log_probs[first_target_id])
        first_target_rank = int(
            np.count_nonzero(first_log_probs > first_target_log_prob) + 1
        )
        result = {
            "evidence": "live_fast_path",
            "generated_text": generated,
            "target": target,
            "target_probability": keep["geometric_mean_probability"],
            "floor_probability": floor["geometric_mean_probability"],
            "score_kind": "geometric_mean_teacher_forced_token_probability",
            "first_target_token": self.fast.tokenizer.decode([first_target_id]),
            "first_target_token_probability": math.exp(first_target_log_prob),
            "first_target_token_rank": first_target_rank,
            "admission": {
                "status": admission.status.value,
                "log_lift_nats": admission.log_lift_nats,
                "probability_ratio": admission.probability_ratio,
                "greedy_match": admission.greedy_match,
                "message": admission.message,
            },
            "elapsed_seconds": keep["elapsed_seconds"] + floor["elapsed_seconds"],
            "memory_prefilled_once": True,
            "memory_input_digest": state["persistent_memory"].input_digest,
        }
        session.results["recall"] = result
        session.engine_state["floor_score"] = floor
        session.phase = DemoPhase.RECALLED
        return result

    def forget(self, session):
        self._require_state(session)
        if session.phase not in (DemoPhase.INGESTED, DemoPhase.RECALLED):
            raise EngineError("ingest_or_recall_required")
        state = session.engine_state
        started = time.perf_counter()
        deleted_memory = self.fast.delete_persistent(
            state["persistent_memory"],
            tuple(state["forget_positions"]),
            kind="single_precision_masked_refit",
        )
        deletion_seconds = time.perf_counter() - started
        state["persistent_deleted"] = deleted_memory
        generated, _ = self.fast.generate_persistent(
            deleted_memory,
            session.audit_probe or "",
        )
        exact = self.fast.score_persistent(
            deleted_memory,
            session.audit_probe or "",
            state["target_ids"],
        )
        floor = state.get("floor_score")
        if floor is None:
            floor = self.fast.score_persistent(
                state["persistent_never"],
                session.audit_probe or "",
                state["target_ids"],
            )
            state["floor_score"] = floor
        fallback_reason = getattr(deleted_memory, "fallback_reason", None)
        result = {
            "evidence": "live_fast_path",
            "method": (
                "full_repack_fallback"
                if fallback_reason
                else "exact_behavioral_drop"
            ),
            "generated_text": generated,
            "before_probability": session.results.get("recall", {}).get(
                "target_probability"
            ),
            "after_probability": exact["geometric_mean_probability"],
            "floor_probability": floor["geometric_mean_probability"],
            "score_kind": "geometric_mean_teacher_forced_token_probability",
            "deletion_ms": deletion_seconds * 1000.0,
            "memory_prefilled_once": True,
            "memory_input_digest": deleted_memory.input_digest,
            "fallback_reason": fallback_reason,
            "message": (
                "The fixed-C deletion budget was exceeded, so the stored "
                "memory was fully repacked without the record."
                if fallback_reason
                else (
                    "One query-independent masked refit was applied to the "
                    "retained prefilled memory; the float64 certificate is a "
                    "separate queued computation."
                )
            ),
        }
        session.results["forget"] = result
        session.phase = DemoPhase.FORGOTTEN
        return result

    def certify(self, session, progress=None):
        self._require_state(session)
        if session.phase not in (
            DemoPhase.FORGOTTEN,
            DemoPhase.ATTACKED,
            DemoPhase.TWIN_TESTED,
        ):
            raise EngineError("forget_required")

        deletion_ranges = tuple(
            CharacterRange(start, end)
            for start, end in (
                session.deletion_ranges
                or ((int(session.secret_start), int(session.secret_end)),)
            )
        )
        selection = SelectedSpan(
            session.memory_text or "",
            deletion_ranges[0].start,
            deletion_ranges[0].end,
            deletion_ranges=deletion_ranges,
            deletion_scope=session.deletion_scope,
            record_id=session.record_id,
        )
        if progress is not None:
            progress(0.03)
        memory, forget_positions = self.certificate.pack(selection, with_fact=True)
        target_ids = self.certificate.target_ids(
            session.engine_state["target_value"], session.audit_probe or ""
        )
        started = time.perf_counter()
        persistent = self.certificate.prefill_persistent(memory)
        if progress is not None:
            progress(0.15)
        overrides, persistent_states = (
            self.certificate.persistent_certificate_states(
                persistent,
                tuple(forget_positions),
            )
        )
        persistent_states["proxy"] = self.certificate.delete_persistent(
            persistent,
            tuple(forget_positions),
            kind="single_precision_masked_refit_bridge",
        )
        if progress is not None:
            progress(0.75)
        distributions: dict[str, np.ndarray] = {}
        scores: dict[str, dict[str, Any]] = {}
        for case_index, case in enumerate(("exact", "refit", "decay", "proxy")):
            score = self.certificate.score_persistent(
                persistent_states[case],
                session.audit_probe or "",
                target_ids,
            )
            scores[case] = score
            distributions[case] = score["first_log_probs"]
            if progress is not None:
                progress(0.78 + 0.04 * case_index)
        p = np.exp(distributions["exact"])
        kl = float(np.sum(p * (distributions["exact"] - distributions["refit"])))
        proxy_kl = float(
            np.sum(p * (distributions["exact"] - distributions["proxy"]))
        )
        p_decay = np.exp(distributions["decay"])
        decay_kl = float(
            np.sum(p_decay * (distributions["decay"] - distributions["refit"]))
        )
        bridge_prompt = elicitation_prompt(session.audit_probe or "", 4)
        bridge_scores = {
            case: self.certificate.score_persistent(
                persistent_states[case],
                bridge_prompt,
                target_ids,
            )
            for case in ("exact", "refit", "proxy")
        }
        classified = classify_certificate(max(0.0, kl))
        result = {
            "evidence": "live_float64_certificate",
            "kl_nats": classified.kl_nats,
            "exact_proxy_kl_nats": max(0.0, proxy_kl),
            "decay_kl_nats": max(0.0, decay_kl),
            "band": classified.band.value,
            "probe_scoped": True,
            "probe": session.audit_probe,
            "deletion_scope": session.deletion_scope,
            "record_id": session.record_id,
            "selected_token_count": len(forget_positions),
            "decrement_fallbacks": overrides["n_fallback"],
            "head_gates": overrides["n_solves"],
            "fixed_c_feasible": overrides["fixed_c_feasible"],
            "max_functional_deviation": overrides["max_functional_deviation"],
            "max_candidate_deviation": overrides["max_candidate_deviation"],
            "functional_tolerance": overrides["functional_tolerance"],
            "used_refit_fallback": overrides["used_refit_fallback"],
            "fallback_details": overrides["fallback_details"],
            "memory_prefilled_once": True,
            "memory_input_digest": persistent.input_digest,
            "query_independent_solve": True,
            "same_context_bridge": {
                "prompt": bridge_prompt,
                "target_probabilities": {
                    case: score["geometric_mean_probability"]
                    for case, score in bridge_scores.items()
                },
                "mechanisms": {
                    "exact": "float64 decrement",
                    "refit": "float64 retained-key refit",
                    "proxy": "single-precision masked refit",
                },
            },
            "elapsed_seconds": time.perf_counter() - started,
            "message": classified.message,
            "exact_target_probability": scores["exact"][
                "geometric_mean_probability"
            ],
            "refit_target_probability": scores["refit"][
                "geometric_mean_probability"
            ],
        }
        session.engine_state["twin_distributions"] = {
            "exact": self.certificate.top_tokens(distributions["exact"]),
            "refit": self.certificate.top_tokens(distributions["refit"]),
        }
        session.results["certificate"] = result
        if progress is not None:
            progress(1.0)
        return result

    def attack(self, session, *, method, kind, budget=None, prompt=None):
        self._require_state(session)
        if session.phase not in (DemoPhase.FORGOTTEN, DemoPhase.ATTACKED):
            raise EngineError("forget_required")
        if method not in ("exact", "icul", "never", "decay"):
            raise EngineError("unsupported_attack_method")
        if kind == "extraction":
            rendered_prompt = extraction_prompt(session.audit_probe or "")
        elif kind == "elicitation":
            rendered_prompt = elicitation_prompt(
                session.audit_probe or "", int(budget or 1)
            )
        elif kind == "freeform":
            if not prompt or not prompt.strip():
                raise EngineError("freeform_prompt_required")
            if len(prompt) > 1_000:
                raise EngineError("freeform_prompt_too_long")
            rendered_prompt = prompt
        else:
            raise EngineError("unsupported_attack_kind")

        state = session.engine_state
        if method == "icul":
            memory = state["persistent_icul"]
        elif method == "never":
            memory = state["persistent_never"]
        elif method == "decay":
            memory = state["persistent_memory"].fork()
            memory.request = GateRequest(
                scale_pos=tuple(state["forget_positions"]),
                scale_factor=DECAY,
                gate_floor=float(
                    getattr(
                        getattr(state["persistent_memory"], "request", None),
                        "gate_floor",
                        0.0,
                    )
                ),
            )
        else:
            memory = state.get("persistent_deleted")
            if memory is None:
                raise EngineError("forget_required")

        generated, _ = self.fast.generate_persistent(memory, rendered_prompt)
        score = self.fast.score_persistent(
            memory,
            rendered_prompt,
            state["target_ids"],
        )
        floor = None
        if method != "never":
            floor = self.fast.score_persistent(
                state["persistent_never"],
                rendered_prompt,
                state["target_ids"],
            )
        floor_probability = (
            score["geometric_mean_probability"]
            if floor is None
            else floor["geometric_mean_probability"]
        )
        ratio = score["geometric_mean_probability"] / max(floor_probability, 1e-300)
        result = {
            "evidence": "live_fast_path",
            "method": method,
            "kind": kind,
            "budget": budget,
            "target_probability": score["geometric_mean_probability"],
            "floor_probability": floor_probability,
            "probability_ratio_to_floor": ratio,
            "at_floor": ratio <= 1.5,
            "generated_text": generated,
            "score_kind": "geometric_mean_teacher_forced_token_probability",
            "memory_prefilled_once": True,
            "memory_input_digest": memory.input_digest,
            "message": "Empirical attack result; this is a stress test, not the certificate.",
        }
        session.results.setdefault("attacks", []).append(result)
        session.phase = DemoPhase.ATTACKED
        return result

    def twin_start(self, session):
        certificate = session.results.get("certificate")
        distributions = session.engine_state.get("twin_distributions")
        if certificate is None or distributions is None:
            raise EngineError("certificate_required")
        import secrets

        deleted_pane = "A" if secrets.randbelow(2) == 0 else "B"
        refit_pane = "B" if deleted_pane == "A" else "A"
        session.engine_state["deleted_pane"] = deleted_pane
        return {
            "evidence": "live_float64_certificate",
            "panes": {
                deleted_pane: {"top_tokens": distributions["exact"]},
                refit_pane: {"top_tokens": distributions["refit"]},
            },
            "scope": "registered_audit_probe",
            "prompt": session.audit_probe,
            "message": "Guess which anonymous next-token distribution came from deletion.",
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "model_id": self.fast.config.model_id,
            "model_revision": getattr(self.fast, "resolved_model_revision", None),
            "recovery_state_restored": (
                getattr(self.fast, "recovery_state", None) is not None
            ),
            "window": self.fast.config.window,
            "memory_copies": self.fast.config.copies,
            "fast": {
                "device": self.fast.config.device,
                "dtype": self.fast.config.dtype,
                "loaded": self.fast.loaded,
                "load_seconds": self.fast.load_seconds,
            },
            "certificate": {
                "device": self.certificate.config.device,
                "dtype": self.certificate.config.dtype,
                "loaded": self.certificate.loaded,
                "load_seconds": self.certificate.load_seconds,
            },
        }

    @staticmethod
    def _require_state(session: SessionRecord) -> None:
        if session.engine_state.get("persistent_memory") is None:
            raise EngineError("ingest_required")

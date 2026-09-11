"""Audit-only Gemma runtime with spawned CPU certificate construction.

The model/MPS/MLX side remains one serialized executor.  Only copied float64
key arrays enter :mod:`gemma_sv.parallel_certificate_v2` workers.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import threading
from typing import Any, Callable, Iterable

from gemma_sv.parallel_certificate_v2 import parallel_certificate_overrides

from .gate_context import GateRequest
from .gemma_engine import (
    DECAY,
    NU,
    GemmaRuntime,
    PersistentGemmaMemory,
    RuntimeConfig,
)


CertificateProgress = Callable[[int, int], None]


class AuditGemmaRuntimeV2(GemmaRuntime):
    """One serialized accelerator runtime plus bounded spawned CPU workers."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        certificate_workers: int = 8,
        certificate_progress: CertificateProgress | None = None,
    ):
        super().__init__(config)
        self._audit_execution_lock = threading.RLock()
        self.certificate_workers = self._validate_workers(certificate_workers)
        self._certificate_progress = certificate_progress

    @staticmethod
    def _validate_workers(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("certificate_workers must be an integer")
        if not 1 <= value <= 16:
            raise ValueError("certificate_workers must be in [1, 16]")
        return value

    def set_certificate_progress_callback(
        self,
        callback: CertificateProgress | None,
    ) -> None:
        if callback is not None and not callable(callback):
            raise ValueError("certificate progress callback must be callable")
        self._certificate_progress = callback

    @contextmanager
    def _persistent_branch(
        self,
        memory: PersistentGemmaMemory,
    ) -> Iterable[PersistentGemmaMemory]:
        with self._audit_execution_lock:
            with super()._persistent_branch(memory) as branch:
                yield branch

    def prefill_persistent(
        self,
        memory_ids: list[int],
        *,
        request: GateRequest | None = None,
    ) -> PersistentGemmaMemory:
        with self._audit_execution_lock:
            return super().prefill_persistent(memory_ids, request=request)

    def delete_persistent(
        self,
        memory: PersistentGemmaMemory,
        forget_positions: tuple[int, ...] | list[int],
        *,
        alpha_by_layer: dict[int, object] | None = None,
        kind: str = "masked_refit",
    ) -> PersistentGemmaMemory:
        with self._audit_execution_lock:
            return super().delete_persistent(
                memory,
                forget_positions,
                alpha_by_layer=alpha_by_layer,
                kind=kind,
            )

    def persistent_certificate_states(
        self,
        memory: PersistentGemmaMemory,
        forget_positions: tuple[int, ...] | list[int],
    ) -> tuple[dict[str, Any], dict[str, PersistentGemmaMemory]]:
        """Build only the exact executed policy state plus compact diagnostics."""

        with self._audit_execution_lock:
            self.ensure_loaded()
            keys = {
                layer_id: session.kf.detach().cpu().double().numpy()
                for layer_id, session in memory.layer_sessions.items()
            }
            layer_ids = sorted(keys)
            if not layer_ids:
                raise RuntimeError("persistent certificate has no global layers")
            n_heads = int(keys[layer_ids[0]].shape[0])
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
            if len(chunks) != 1 or len(boundary_modes) != 1:
                raise RuntimeError(
                    "persistent certificate requires one chunk and box mode"
                )
            box_values = {
                layer_id: float(memory.layer_sessions[layer_id].box_C)
                for layer_id in layer_ids
            }
            box_maps = {
                layer_id: {
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
                layer_id: float(memory.layer_sessions[layer_id].kpar)
                for layer_id in layer_ids
            }
            per_boundary_box = next(iter(boundary_modes))
            frozen_box = box_values[layer_ids[0]]
            frozen_boxes = (
                box_maps[layer_ids[0]] if per_boundary_box else None
            )
            if per_boundary_box:
                if not frozen_boxes or any(
                    box_maps[layer_id] != frozen_boxes
                    for layer_id in layer_ids
                ):
                    raise RuntimeError(
                        "persistent certificate boundary boxes differ "
                        "across layers"
                    )
            elif any(
                abs(value - frozen_box) > 1e-12
                for value in box_values.values()
            ):
                raise RuntimeError(
                    "persistent certificate requires one frozen global C"
                )

            overrides = parallel_certificate_overrides(
                keys,
                layer_ids,
                forget_positions,
                memory.token_count,
                n_heads,
                nu=getattr(self, "resolved_nu", NU),
                chunk=next(iter(chunks)),
                decay=DECAY,
                box_C=None if per_boundary_box else frozen_box,
                box_C_by_boundary=frozen_boxes,
                kpar_by_layer=kpars,
                workers=self.certificate_workers,
                backend="process",
                progress=self._certificate_progress,
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
                        "certificate boundary boxes differ from prefill"
                    )
            elif abs(float(overrides["box_C"]) - frozen_box) > 1e-12:
                raise RuntimeError(
                    "certificate C differs from the persistent prefill C"
                )

            exact_alpha = overrides.pop("exact")
            # Refit coefficients were required inside the solver for functional
            # conformance.  Decay coefficients preserve the shared solver
            # schema, but neither array set needs a PersistentGemmaMemory here.
            overrides.pop("refit")
            overrides.pop("decay")
            exact = self.delete_persistent(
                memory,
                forget_positions,
                alpha_by_layer=exact_alpha,
                kind="float64_exact_policy",
            )
            overrides["materialization"] = {
                "persistent_states": ["exact_policy"],
                "refit_coefficients_used_for_solver_conformance": True,
                "refit_persistent_state_constructed": False,
                "decay_persistent_state_constructed": False,
            }
            return overrides, {
                "exact": exact,
            }

    def synchronize_accelerator(self) -> None:
        """Synchronize completed record work without changing live tensors."""

        try:
            import torch
        except ImportError:
            return
        if self.config.device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    def cleanup_record_boundary(self) -> None:
        """Release unreachable record objects only between full records."""

        with self._audit_execution_lock:
            self.synchronize_accelerator()
            gc.collect()
            try:
                import torch
            except ImportError:
                return
            if self.config.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()

    @contextmanager
    def record_boundary(
        self,
        _slot: int,
        _record_id: str,
    ) -> Iterable[None]:
        """Clean before/after a record, never during its scientific work."""

        self.cleanup_record_boundary()
        try:
            yield
        finally:
            self.set_certificate_progress_callback(None)
            self.cleanup_record_boundary()


__all__ = ["AuditGemmaRuntimeV2", "CertificateProgress"]

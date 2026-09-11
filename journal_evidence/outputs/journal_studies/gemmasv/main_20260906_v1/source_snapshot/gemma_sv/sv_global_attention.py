"""SVGlobalAttention: a drop-in for ``Gemma3Attention`` on the GLOBAL (non-sliding)
layers, whose readout is the certified Support-Vector gate instead of softmax.

It reuses the *base* ``Gemma3Attention``'s projections (q/k/v/o), QK-norm, RoPE and
GQA config unchanged -- only the attention operator (softmax over q.k) is replaced
by the SV-gated chunk-frozen causal readout from ``svattn.causal_sv_attention``.

Math fidelity to the companion SV-Attention paper is enforced, not assumed:
  * Gate: the one-class SVDD over the keys with box ``C = 1/(nu*n)``;
    ``nu`` is the budget knob, ``C`` overridable. Any such C is a feasible box, so
    the partition (S, E, R) and its certificate are well-defined.
  * Certificate (complementary slackness): a reserve token (alpha=0)
    contributes EXACTLY zero to the readout. Preserved here -- the hybrid "softmax"
    readout multiplies the learned q.k weights on the long-range prefix by alpha, so
    alpha=0 -> weight 0 (recent local tokens keep alpha=1, by design).
  * The deployed readout is a hybrid (local q.k + alpha-gated prefix), not the
    capability paper's pure RBF average -- a deliberate choice so grafted Gemma
    stays competitive (the pure gate plateaus as an LM layer). Set readout="rbf" for
    the literal capability-paper readout.
  * EXACT forgetting is NOT done here: the forward uses the single-precision FISTA
    solver (training/inference). The unlearning *guarantee* runs the float64 C&P
    decrement in ``gemma_sv.unlearn`` -- never FISTA.

Smoke-tested via ``gemma_sv.smoke_test`` (random-init Gemma3, no gated download):
layer detection, graft, a finite forward (RoPE-on-keys + GQA expand + base scaling
+ batched FISTA gate), and the state-exact float64 forget certificate all pass.
Open design choices still exposed as flags (need an ablation): ``rope_on_keys``
(RBF ||q-k||^2 is not RoPE-invariant) and ``chunk`` (gate granularity).
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SVRuntimeState:
    """Request-scoped gate controls used by inference and certificate forwards."""

    alpha_override: object = None
    drop_pos: tuple[int, ...] | None = None
    scale_pos: tuple[int, ...] | None = None
    scale_factor: float = 1.0
    gate_floor: float = 0.0


@dataclass
class SVDecodeSession:
    """Per-generation cache for incremental decoding on a grafted layer.

    Holds the RoPE'd, GQA-expanded keys/values seen so far plus the solved
    chunk-boundary gates. The RBF bandwidth is frozen at prefill: the median
    heuristic subsamples 256 keys, so one extra token cannot move it beyond
    its own sampling noise, and freezing keeps decode steps solver-free until
    the next chunk boundary.
    """

    kf: object = None            # (G, T, d) torch tensor
    vf: object = None            # (G, T, d) torch tensor
    gates: dict = None           # {chunk_start: (alpha, valid)}
    kpar: float = 0.0
    box_C: float = 0.0           # frozen at persistent-memory prefill
    box_C_by_boundary: dict = None
    frozen: bool = False         # do not admit subsequent query tokens
    frozen_boundary: int = 0     # long-range prefix used by every query
    batch: int = 0

    def clone(self) -> "SVDecodeSession":
        def duplicate(value):
            return value.clone() if hasattr(value, "clone") else copy.deepcopy(value)

        return SVDecodeSession(
            kf=None if self.kf is None else duplicate(self.kf),
            vf=None if self.vf is None else duplicate(self.vf),
            gates={
                boundary: (duplicate(alpha), duplicate(valid))
                for boundary, (alpha, valid) in (self.gates or {}).items()
            },
            kpar=float(self.kpar),
            box_C=float(self.box_C),
            box_C_by_boundary={
                int(boundary): float(value)
                for boundary, value in (self.box_C_by_boundary or {}).items()
            },
            frozen=bool(self.frozen),
            frozen_boundary=int(self.frozen_boundary),
            batch=int(self.batch),
        )


class SVGlobalAttention(nn.Module):
    def __init__(self, base_attn: nn.Module, *, nu: float = 0.3,
                 C: Optional[float] = None, kpar: Optional[float] = None,
                 chunk: int = 128, gate: bool = True, rope_on_keys: bool = True,
                 solver: str = "mlx", fista_iters: int = 80,
                 solver_seed: int = 0,
                 partition_tol: float = 1e-3,
                 readout: str = "softmax", normalize: bool = True,
                 preserve_prefix_mass: bool = False,
                 per_boundary_box: bool = False):
        super().__init__()
        self.base = base_attn               # keeps q/k/v/o proj, q/k norm, RoPE, GQA cfg
        # Stay discoverable by layer_select after the graft: a grafted layer is still
        # a GLOBAL (non-sliding) layer, so advertise is_sliding=False + carry layer_idx.
        # Without this, find_global_attention_layers returns [] post-graft, breaking
        # extract_global_layer_keys and the stage-1 attention-transfer hooks.
        self.is_sliding = False
        self.layer_idx = getattr(base_attn, "layer_idx", -1)
        self.nu = float(nu)                 # budget; box C = 1/(nu*n)
        self.C = None if C is None else float(C)   # explicit box override
        self.kpar = kpar                    # None -> median pairwise-distance heuristic
        self.chunk = int(chunk)
        self.gate = bool(gate)
        self.rope_on_keys = bool(rope_on_keys)
        self.solver = str(solver)
        self.fista_iters = int(fista_iters)
        self.solver_seed = int(solver_seed)
        self.partition_tol = float(partition_tol)
        if not 0 < self.partition_tol < 1:
            raise ValueError("partition_tol must be in (0, 1)")
        self.readout = str(readout)
        self.normalize = bool(normalize)
        # Raw SVDD coefficients sum to one, while recent-token multipliers are
        # one per token. This optional readout restores the prefix's pre-gate
        # softmax mass; reserve coefficients remain exactly zero.
        self.preserve_prefix_mass = bool(preserve_prefix_mass)
        self.per_boundary_box = bool(per_boundary_box)
        # Stage-1 (attention transfer) makes the RBF bandwidth trainable. Off by default
        # (inference/forget use the median heuristic); enable_learnable_kernel() flips it
        # on and registers log_kpar so an optimizer can adapt the gate to each layer.
        self._learn_kpar = False
        # Optional {chunk_start: (G, start)} exact gate alphas to USE instead of the FISTA
        # solve -- set by the model-output-level unlearning demo to thread the float64 C&P
        # (decrement / refit-without) gate through the live forward. None = normal inference.
        self._alpha_override = None
        # Optional list of prefix positions to FORGET/EVICT from the long-range memory: they
        # leave the gate fit and contribute zero to the readout (== refit-without at FISTA
        # precision). Set by the in-context-unlearning eval. None = normal inference.
        self._drop_pos = None
        # Optional DECAY baseline: scale these prefix positions' readout weight by _scale_factor
        # (kept in the gate, just downweighted -> leaves residual). The approximate-unlearning foil.
        self._scale_pos = None
        self._scale_factor = 1.0
        # Query-time-only dense retained-key residual.  A value f interpolates
        # the gate multiplier as f + (1-f)*alpha. Deleted positions are still
        # hard-masked downstream. Full prefill remains on the exact SV path.
        self._gate_floor = 0.0
        # Incremental-decoding session (see SVDecodeSession). None = full recompute.
        self._decode_session = None

    def runtime_state(self) -> SVRuntimeState:
        """Snapshot mutable forward controls without copying large alpha arrays."""

        return SVRuntimeState(
            alpha_override=self._alpha_override,
            drop_pos=None if self._drop_pos is None else tuple(self._drop_pos),
            scale_pos=None if self._scale_pos is None else tuple(self._scale_pos),
            scale_factor=float(self._scale_factor),
            gate_floor=float(self._gate_floor),
        )

    def set_runtime_state(self, state: SVRuntimeState | None = None) -> None:
        """Apply request controls; ``None`` restores the neutral inference state."""

        state = state or SVRuntimeState()
        self._alpha_override = state.alpha_override
        self._drop_pos = None if state.drop_pos is None else list(state.drop_pos)
        self._scale_pos = None if state.scale_pos is None else list(state.scale_pos)
        self._scale_factor = float(state.scale_factor)
        self._gate_floor = float(state.gate_floor)
        if not 0.0 <= self._gate_floor <= 1.0:
            raise ValueError("gate_floor must be in [0, 1]")

    @contextmanager
    def use_runtime_state(self, state: SVRuntimeState | None = None):
        """Temporarily apply gate controls and restore them even after failure.

        A caller serving a shared model must still hold one model-wide lock around
        the complete forward.  This context manager makes state restoration
        exception-safe; it does not claim that concurrent forwards are safe.
        """

        previous = self.runtime_state()
        self.set_runtime_state(state)
        try:
            yield self
        finally:
            self.set_runtime_state(previous)

    def enable_learnable_kernel(self, kpar_init: float) -> None:
        """Stage-1: register the RBF bandwidth as a trainable ``nn.Parameter`` (stored
        in log-space to stay positive), initialised at ``kpar_init`` (typically each
        layer's median key distance). Gradients reach it via the gate's implicit VJP.
        The box ``C = 1/(nu*n)`` is left principled -- it sets the budget/threshold and
        the certificate, and entangles with the discrete gate structure; learning it is
        a deferred refinement."""
        dev = next(self.base.parameters()).device
        init = torch.tensor(float(kpar_init), device=dev).clamp_min(1e-6).log()
        self.log_kpar = nn.Parameter(init)
        self._learn_kpar = True

    def _box_C(self, n: int) -> float:
        """Box ``C = 1/(nu*n)``, with an optional explicit override."""
        if self.C is not None:
            return self.C
        return float(min(1.0, 1.0 / (self.nu * max(n, 1))))

    def _box_spec(self, total_tokens: int):
        """Return one legacy global box or one canonical box per prefix solve."""
        if not self.per_boundary_box:
            return self._box_C(total_tokens)
        return {
            start: self._box_C(start)
            for start in range(self.chunk, total_tokens, self.chunk)
        }

    def _kpar_for(self, keys: torch.Tensor) -> float:
        """RBF bandwidth: explicit ``self.kpar`` or the median pairwise key distance
        (the p3b heuristic), as a detached scalar."""
        if self.kpar is not None:
            return float(self.kpar)
        with torch.no_grad():
            x = keys.reshape(-1, keys.shape[-1])
            if x.shape[0] > 256:
                index = torch.linspace(
                    0,
                    x.shape[0] - 1,
                    steps=256,
                    device=x.device,
                ).round().long()
                x = x[index]
            d2 = torch.cdist(x, x) ** 2
            m = d2[d2 > 0].median()
            return float(m.sqrt().clamp_min(1e-6))

    def _kpar_value(self, keys: torch.Tensor):
        """The bandwidth used by the readout: a differentiable ``log_kpar.exp()`` tensor
        when stage-1 training is enabled, else the detached float heuristic."""
        if self._learn_kpar:
            return self.log_kpar.exp()
        return self._kpar_for(keys)

    def _project_qkv(self, hidden_states: torch.Tensor,
                     position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        """Replicate ``Gemma3Attention``'s projection path: q/k/v proj -> head split
        -> QK-norm -> (optional) RoPE -> GQA expand. Returns q, k, v each
        (B, H, T, d). Single source of truth so the readout (forward) and the
        unlearning key extraction (gemma_sv.unlearn) never drift."""
        from transformers.models.gemma3.modeling_gemma3 import (
            apply_rotary_pos_emb, repeat_kv)
        b = self.base
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, b.head_dim)
        q = b.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = b.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = b.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q = b.q_norm(q)
        k = b.k_norm(k)
        if self.rope_on_keys and position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, b.num_key_value_groups)
        v = repeat_kv(v, b.num_key_value_groups)
        return q, k, v

    def begin_decode_session(self) -> None:
        """Cache keys/values/gates across decode steps until the session ends.

        Only valid for the hybrid ``readout="softmax"`` graft. The caller must
        hold the model gate lock for the whole session and keep the runtime
        state (drop/scale/alpha controls) fixed within it.
        """
        if self.readout != "softmax":
            raise RuntimeError("decode sessions require the softmax readout")
        self._decode_session = SVDecodeSession(gates={})

    def end_decode_session(self) -> None:
        self._decode_session = None

    def snapshot_decode_session(self) -> SVDecodeSession:
        """Copy the prefilled long-range memory for query-independent reuse."""

        if self._decode_session is None:
            raise RuntimeError("no decode session to snapshot")
        return self._decode_session.clone()

    def restore_decode_session(self, session: SVDecodeSession) -> None:
        """Install an isolated copy of a previously prefilled memory."""

        if self.readout != "softmax":
            raise RuntimeError("decode sessions require the softmax readout")
        self._decode_session = session.clone()

    def _decode_step(self, hidden_states, position_embeddings, input_shape):
        """One cached decode step: project the new token, extend the cache,
        solve at most one new chunk-boundary gate, and read out one query."""
        from svattn.causal_sv_attention import (
            compute_boundary_gates, sv_softmax_decode_step)

        session = self._decode_session
        q, k, v = self._project_qkv(hidden_states, position_embeddings)
        B, H, T_new, d = q.shape
        G = B * H
        session.kf = torch.cat([session.kf, k.reshape(G, T_new, d)], dim=1)
        session.vf = torch.cat([session.vf, v.reshape(G, T_new, d)], dim=1)
        T = session.kf.shape[1]
        boundary = (
            session.frozen_boundary
            if session.frozen
            else ((T - 1) // self.chunk) * self.chunk
        )
        if self.per_boundary_box:
            box_C = (session.box_C_by_boundary or {}).get(
                boundary,
                self._box_C(boundary),
            )
        else:
            box_C = session.box_C or self._box_C(T)
        thresh = int(torch.ceil(torch.tensor(1.0 / box_C)).item()) + 2
        if (not session.frozen and self.gate and boundary >= thresh
                and boundary not in session.gates):
            override = (
                self._alpha_override
                if self._alpha_override is not None
                and boundary in self._alpha_override
                else None
            )
            session.gates.update(compute_boundary_gates(
                session.kf, box_C, session.kpar, self.chunk,
                gate=self.gate, fista_iters=self.fista_iters,
                tol=self.partition_tol,
                alpha_override=override, drop_pos=self._drop_pos,
                boundaries=[boundary],
                solver_seed=self.solver_seed))
            session.box_C_by_boundary = dict(
                session.box_C_by_boundary or {}
            )
            session.box_C_by_boundary[boundary] = box_C
        of = sv_softmax_decode_step(
            session.kf, session.vf, q.reshape(G, T_new, d), session.gates,
            self.chunk, scaling=self.base.scaling, normalize=self.normalize,
            drop_pos=self._drop_pos, scale_pos=self._scale_pos,
            scale_factor=self._scale_factor,
            gate_floor=self._gate_floor,
            preserve_prefix_mass=self.preserve_prefix_mass,
            boundary_override=boundary if session.frozen else None)
        attn_output = (of.reshape(B, H, T_new, d).transpose(1, 2)
                       .reshape(*input_shape, -1).contiguous())
        return self.base.o_proj(attn_output), None

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values=None, **kwargs):
        from svattn.causal_sv_attention import causal_sv_readout_mlx_batched

        b = self.base
        input_shape = hidden_states.shape[:-1]
        session = self._decode_session
        if session is not None and session.kf is not None:
            return self._decode_step(hidden_states, position_embeddings, input_shape)

        q, k, v = self._project_qkv(hidden_states, position_embeddings)
        B, H, T, d = q.shape
        G = B * H
        Qf, Kf, Vf = q.reshape(G, T, d), k.reshape(G, T, d), v.reshape(G, T, d)

        # Certificate preserved: the readout weights the long-range prefix by the SVDD
        # alpha, so a reserve token (alpha=0) contributes EXACTLY zero. Fidelity: the
        # base layer's own scaling (Gemma3 = query_pre_attn_scalar**-0.5) is threaded
        # into the q.k softmax path (not 1/sqrt(d)) so the gate stands in faithfully for
        # the original global attention. attention_mask unused (the chunk-frozen readout
        # is causal by construction).
        kpar = self._kpar_value(Kf)
        box_spec = self._box_spec(T)
        gate_sink = {} if session is not None else None
        of = causal_sv_readout_mlx_batched(
            Kf, Vf, Qf, C=box_spec, kpar=kpar, chunk=self.chunk,
            normalize=self.normalize, gate=self.gate,
            fista_iters=self.fista_iters, tol=self.partition_tol,
            readout=self.readout, scaling=b.scaling,
            alpha_override=self._alpha_override, drop_pos=self._drop_pos,
            scale_pos=self._scale_pos, scale_factor=self._scale_factor,
            gate_sink=gate_sink,
            preserve_prefix_mass=self.preserve_prefix_mass,
            solver_seed=self.solver_seed)
        if session is not None:
            # Prefill: seed the decode cache with keys/values/gates/bandwidth.
            session.kf, session.vf = Kf, Vf
            session.gates = gate_sink
            session.kpar = float(kpar)
            session.box_C_by_boundary = (
                {
                    int(start): float(value)
                    for start, value in box_spec.items()
                }
                if isinstance(box_spec, dict)
                else {
                    int(start): float(box_spec)
                    for start in gate_sink
                }
            )
            session.frozen_boundary = max(gate_sink, default=0)
            session.box_C = float(
                session.box_C_by_boundary.get(
                    session.frozen_boundary,
                    self._box_C(T),
                )
            )
            session.batch = B

        attn_output = (of.reshape(B, H, T, d).transpose(1, 2)
                       .reshape(*input_shape, -1).contiguous())
        attn_output = b.o_proj(attn_output)
        return attn_output, None

"""Causal, chunk-frozen, multi-head Support Vector Attention.

Chunk-frozen contract: the sequence is split into chunks of length L, and the
queries in chunk t read against the gate fit on the *causal prefix* (all tokens
strictly before chunk t). No query sees its own or a future chunk. Within a
chunk the gate is frozen (computed once per chunk, not per token); a
token-by-token maintained update is retained as a verification reference.

The gate is the differentiable one-class SVDD over the prefix keys
(`fast_diff_svdd`): reserve tokens receive zero weight for the solved prefix,
and alpha is differentiable via the implicit VJP, so the max-margin solve is
load-bearing in training. Until the prefix is large enough for a
well-posed gate (n >= ceil(1/C) + 2) -- or if the gate is degenerate (empty
margin set) -- the chunk falls back to a uniform-weight RBF readout (warmup).
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import torch
import torch.nn as nn

from cp_svm.oneclass_fast import FastOneClassSVM
from .fast_diff_svdd import (fast_diff_svdd, diff_svdd_from_solver,
                             diff_svdd_from_partition, batched_svdd_from_masks)
from .sv_attention import rbf_gram


def _gate_weights(Xpref: torch.Tensor, Qc: torch.Tensor, C: float, kpar: float,
                  gate: bool = True):
    """(chunk, n_pref) weights = kappa(Qc, Xpref), optionally gated by the SVDD
    alpha. With gate=False (ablation) or a not-yet-well-posed / degenerate gate,
    returns the ungated uniform-RBF weights.
    """
    n = Xpref.shape[0]
    w_kernel = rbf_gram(Qc, Xpref, kpar)                 # (chunk, n_pref)
    if not gate or n < int(math.ceil(1.0 / C)) + 2:
        return w_kernel                                  # ablation / warmup
    try:
        alpha, _ = fast_diff_svdd(Xpref, C=C, kpar=kpar)
    except RuntimeError:
        return w_kernel                                  # degenerate gate
    return w_kernel * alpha.unsqueeze(0)


def _read(Xpref, Vpref, Qc, C, kpar, normalize, gate=True):
    w = _gate_weights(Xpref, Qc, C, kpar, gate)
    O = w @ Vpref
    if normalize:
        O = O / w.sum(1, keepdim=True).clamp_min(1e-8)
    return O


def causal_sv_readout(Kp: torch.Tensor, Vp: torch.Tensor, Qp: torch.Tensor,
                      C: float, kpar: float, chunk: int, normalize: bool = True,
                      exact_sequential: bool = False, gate: bool = True):
    """Causal readout for one (key, value, query) stream. Returns (T, dv).

    chunk-frozen (default): queries in each chunk read the prefix before the
    chunk -- one gate solve per chunk. exact_sequential: query t reads the prefix
    [0, t) -- a gate update per token used as a slow verification reference.
    """
    T, dv = Kp.shape[0], Vp.shape[1]
    out = []
    if exact_sequential:
        for t in range(T):
            if t == 0:
                out.append(Qp[0:1].new_zeros(1, dv))
            else:
                out.append(_read(Kp[:t], Vp[:t], Qp[t:t + 1], C, kpar, normalize, gate))
        return torch.cat(out, dim=0)
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        if start == 0:
            out.append(Qp[start:end].new_zeros(end - start, dv))
            continue
        out.append(_read(Kp[:start], Vp[:start], Qp[start:end], C, kpar, normalize, gate))
    return torch.cat(out, dim=0)


def causal_sv_readout_persistent(Kp: torch.Tensor, Vp: torch.Tensor, Qp: torch.Tensor,
                                 C: float, kpar: float, chunk: int, normalize: bool = True,
                                 gate: bool = True, refactor_every: int = 500,
                                 state_sink: list | None = None):
    """Chunk-frozen causal readout with a *persistent* maintained solver.

    Identical contract and (to fp tolerance) identical output as the chunk-frozen
    branch of `causal_sv_readout`, but the one-class gate is maintained across
    chunks by rank-1 Sherman-Morrison updates instead of re-streaming the whole
    prefix every chunk. The prefix-gate solve goes from O(T^2/chunk) point-adds to
    O(T) total. The differentiable alpha is still produced by the implicit-VJP
    `_FastSVDDFn`, snapshotting the maintained partition at each chunk boundary.
    """
    T, dv = Kp.shape[0], Vp.shape[1]
    thresh = int(math.ceil(1.0 / C)) + 2
    Knp = Kp.detach().cpu().numpy().astype(np.float64)
    solver: FastOneClassSVM | None = None
    fed = 0

    def ensure_fed(n: int):
        nonlocal solver, fed
        if n <= fed:
            return
        if solver is None:
            n_seed = max(2, min(thresh, n))
            solver = FastOneClassSVM(C=float(C), ktype="r", kpar=float(kpar),
                                     refactor_every=refactor_every)
            solver.seed_from_qp(Knp[:n_seed])
            fed = n_seed
        while fed < n:
            solver.add_point(Knp[fed])
            fed += 1

    out = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        if start == 0:
            out.append(Qp[start:end].new_zeros(end - start, dv))
            continue
        Xpref, Vpref, Qc = Kp[:start], Vp[:start], Qp[start:end]
        w_kernel = rbf_gram(Qc, Xpref, kpar)
        if not gate or start < thresh:
            w = w_kernel                                 # warmup / ablation
        else:
            ensure_fed(start)
            if len(solver.S) == 0:
                w = w_kernel                             # degenerate gate
            else:
                alpha, _ = diff_svdd_from_solver(Xpref, solver, C, kpar)
                w = w_kernel * alpha.unsqueeze(0)
        O = w @ Vpref
        if normalize:
            O = O / w.sum(1, keepdim=True).clamp_min(1e-8)
        out.append(O)
    if state_sink is not None and solver is not None:
        # effective state = #tokens with nonzero alpha (active support |S u E|)
        state_sink.append(len(solver.S) + len(solver.E))
    return torch.cat(out, dim=0)


def _batched_rbf(A: torch.Tensor, B: torch.Tensor, kpar: float) -> torch.Tensor:
    """Batched RBF gram: A (G, m, d), B (G, n, d) -> (G, m, n), exp(-d^2/kpar^2)."""
    d2 = torch.cdist(A, B) ** 2
    return torch.exp(-d2 / (kpar * kpar))


def _raw_fista_gate(
    alpha_values: np.ndarray,
    reference: torch.Tensor,
    box_C: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert projected FISTA iterates and validate their primal feasibility."""

    alpha = torch.as_tensor(
        np.ascontiguousarray(alpha_values),
        dtype=reference.dtype,
        device=reference.device,
    )
    mass = alpha.sum(dim=1)
    valid = (
        torch.isfinite(alpha).all(dim=1)
        & (alpha.min(dim=1).values >= -1e-6)
        & (alpha.max(dim=1).values <= float(box_C) + 1e-6)
        & ((mass - 1.0).abs() <= 1e-5)
    )
    return alpha, valid


def _boundary_box(C: float | Mapping[int, float], start: int) -> float:
    value = float(C[start]) if isinstance(C, Mapping) else float(C)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"invalid box C at boundary {start}: {value!r}")
    return value


def compute_boundary_gates(
                           Kf: torch.Tensor,
                           C: float | Mapping[int, float],
                           kpar: float,
                           chunk: int,
                           *, gate: bool = True, fista_iters: int = 150,
                           tol: float = 1e-3, state_sink: list | None = None,
                           alpha_override: dict | None = None,
                           drop_pos: list | None = None,
                           boundaries: list[int] | None = None,
                           raw_fista_alpha: bool = False,
                           solver_seed: int | None = None) -> dict:
    """Solve the chunk-boundary SVDD gates for (G, T, d) keys.

    Returns {start: (alpha (G, start) tensor, valid (G,) bool tensor)}.
    ``C`` may be one legacy global box or a mapping from boundary to the
    canonical per-solve box. The latter solves each boundary separately because
    its capped simplex differs. A boundary whose problem is degenerate reports
    ``valid=False`` and callers fall back to the ungated readout.
    ``raw_fista_alpha`` is inference-only: it returns the feasible projected
    iterate directly instead of rebuilding alpha from a ridge-regularized KKT
    partition. This is the residual-measured FP32 deletion proxy path.
    """
    from .mlx_svdd import svdd_fista_mlx, _HAVE_MLX
    if not _HAVE_MLX:
        raise ImportError("solver='mlx' needs the mlx package (Apple silicon).")
    import mlx.core as mx

    G, T, _ = Kf.shape
    starts = boundaries if boundaries is not None else list(range(chunk, T, chunk))
    boxes = {s: _boundary_box(C, s) for s in starts}
    gated = [
        s
        for s in starts
        if gate and s >= int(math.ceil(1.0 / boxes[s])) + 2
    ]

    # alpha_override: caller supplies maintained fp64 gate alphas per boundary
    # ({start: (G, start)}) and skips FISTA. This supports frozen-state
    # decision/readout audits without transferring the batched-path contract.
    if alpha_override is not None:
        result = {}
        for s in gated:
            if s not in alpha_override:
                continue
            alpha = torch.as_tensor(np.asarray(alpha_override[s]),
                                    dtype=Kf.dtype, device=Kf.device)
            result[s] = (alpha, torch.ones(alpha.shape[0], dtype=torch.bool,
                                           device=Kf.device))
        return result

    # Per-boundary boxes define independent QPs.  Solve and materialize one
    # boundary at a time so long contexts do not retain the cubic aggregate
    # ``sum_s G*s*s`` Gram storage.  The global-C compatibility path below
    # still batches boundaries because it shares one capped simplex.
    if isinstance(C, Mapping):
        result = {}
        for s in gated:
            if solver_seed is not None:
                mx.random.seed(int(solver_seed) + int(s))
            box = boxes[s]
            part_tol = tol * box
            alpha_rows = []
            valid_rows = []
            active_counts = []
            # A final 10k-token Gram is several GiB when all heads are batched.
            # Heads are independent QPs, so stream them without changing the
            # solve, RNG order, or fixed-C contract.
            for head in range(G):
                keys = Kf[head : head + 1, :s]
                gram = _batched_rbf(keys, keys, kpar)
                gram_np = gram.detach().to(torch.float32).cpu().numpy()
                mask_np = np.ones((1, s), dtype=np.float32)
                if drop_pos:
                    dp = [d for d in drop_pos if d < s]
                    if dp:
                        mask_np[:, dp] = 0.0
                a_b = np.asarray(
                    svdd_fista_mlx(
                        mx.array(gram_np),
                        box,
                        iters=fista_iters,
                        mask=mx.array(mask_np),
                    )
                )
                active = (
                    ((a_b > part_tol) & (a_b < box - part_tol))
                    | (a_b >= box - part_tol)
                )
                active_counts.append(float(active.sum(1)[0]))
                if raw_fista_alpha:
                    alpha, valid = _raw_fista_gate(a_b, keys, box)
                else:
                    s_mask = torch.from_numpy(
                        np.ascontiguousarray(
                            (a_b > part_tol) & (a_b < box - part_tol)
                        )
                    ).to(Kf.device)
                    e_mask = torch.from_numpy(
                        np.ascontiguousarray(a_b >= box - part_tol)
                    ).to(Kf.device)
                    alpha, valid = batched_svdd_from_masks(
                        gram,
                        box,
                        s_mask,
                        e_mask,
                    )
                alpha_rows.append(alpha)
                valid_rows.append(valid)
                del keys, gram, gram_np, mask_np, a_b
                mx.clear_cache()
            result[s] = (
                torch.cat(alpha_rows, dim=0),
                torch.cat(valid_rows, dim=0),
            )
            if state_sink is not None and s == gated[-1]:
                state_sink.append(float(np.mean(active_counts)))
        return result

    grams = {s: _batched_rbf(Kf[:, :s], Kf[:, :s], kpar) for s in gated}
    alpha_np = {}
    if gated:
        n_max, nb = max(gated), len(gated)
        Kpad = np.zeros((nb * G, n_max, n_max), dtype=np.float32)
        Mpad = np.zeros((nb * G, n_max), dtype=np.float32)
        for i, s in enumerate(gated):
            Kpad[i * G:(i + 1) * G, :s, :s] = (
                grams[s].detach().to(torch.float32).cpu().numpy()
            )
            Mpad[i * G:(i + 1) * G, :s] = 1.0
            if drop_pos:
                dp = [d for d in drop_pos if d < s]
                if dp:
                    Mpad[i * G:(i + 1) * G, dp] = 0.0
        if solver_seed is not None:
            mx.random.seed(int(solver_seed))
        a_all = np.asarray(
            svdd_fista_mlx(
                mx.array(Kpad),
                float(C),
                iters=fista_iters,
                mask=mx.array(Mpad),
            )
        )
        for i, s in enumerate(gated):
            alpha_np[s] = a_all[i * G:(i + 1) * G, :s]
        if state_sink is not None:                      # effective state at last boundary
            last = max(gated)
            box = boxes[last]
            part_tol = tol * box if isinstance(C, Mapping) else tol
            a_last = alpha_np[last]
            active = (
                ((a_last > part_tol) & (a_last < box - part_tol))
                | (a_last >= box - part_tol)
            )
            state_sink.append(float(active.sum(1).mean()))

    result = {}
    for s in gated:
        a_b = alpha_np.get(s)
        if a_b is None:
            continue
        box = boxes[s]
        part_tol = tol * box if isinstance(C, Mapping) else tol
        if raw_fista_alpha:
            result[s] = _raw_fista_gate(a_b, Kf, box)
            continue
        s_mask = torch.from_numpy(
            np.ascontiguousarray(
                (a_b > part_tol) & (a_b < box - part_tol)
            )
        ).to(Kf.device)
        e_mask = torch.from_numpy(
            np.ascontiguousarray(a_b >= box - part_tol)
        ).to(Kf.device)
        result[s] = batched_svdd_from_masks(
            grams[s],
            box,
            s_mask,
            e_mask,
        )
    return result


def _apply_prefix_gate(
    weights: torch.Tensor,
    factors: torch.Tensor,
    prefix_length: int,
    *,
    preserve_prefix_mass: bool,
) -> torch.Tensor:
    """Apply per-key gate factors without unintentionally shrinking the prefix.

    SVDD coefficients sum to one, while the ungated recent-window factors are
    one per token.  Multiplying by raw coefficients therefore makes the entire
    stored prefix fade as it grows.  The mass-preserving path restores the
    prefix's pre-gate softmax mass per query row.  Exact-zero factors remain
    exact zero, so reserve-key inertness is unchanged.
    """
    gated = weights * factors.unsqueeze(-2)
    if not preserve_prefix_mass or prefix_length <= 0:
        return gated

    before = weights[..., :prefix_length].sum(dim=-1, keepdim=True)
    after = gated[..., :prefix_length].sum(dim=-1, keepdim=True)
    scale = before / after.clamp_min(1e-12)
    return torch.cat(
        (
            gated[..., :prefix_length] * scale,
            gated[..., prefix_length:],
        ),
        dim=-1,
    )


def sv_softmax_decode_step(Kf: torch.Tensor, Vf: torch.Tensor, q: torch.Tensor,
                           gates: dict, chunk: int, *, scaling: float | None = None,
                           normalize: bool = True, drop_pos: list | None = None,
                           scale_pos: list | None = None,
                           scale_factor: float = 1.0,
                           gate_floor: float = 0.0,
                           preserve_prefix_mass: bool = False,
                           boundary_override: int | None = None) -> torch.Tensor:
    """Hybrid softmax readout for ONE new query at absolute position T-1.

    Kf, Vf: (G, T, d) keys/values including the new token; q: (G, 1, d).
    ``gates`` maps chunk starts to (alpha, valid) as computed by
    :func:`compute_boundary_gates`. Reproduces exactly one query row of the
    ``readout="softmax"`` branch of :func:`causal_sv_readout_mlx_batched`.
    ``boundary_override`` freezes the long-range prefix for persistent-memory
    queries, so query tokens are not admitted into the stored gate.
    """
    G, T, _ = Kf.shape
    inv_scale = (1.0 / math.sqrt(Kf.shape[-1])) if scaling is None else float(scaling)
    start = (
        ((T - 1) // chunk) * chunk
        if boundary_override is None
        else int(boundary_override)
    )
    scores = q @ Kf.transpose(-1, -2) * inv_scale            # (G, 1, T) all past: no mask
    w = torch.softmax(scores, dim=-1)
    entry = gates.get(start)
    if entry is not None:
        alpha, gate_valid = entry
        floor = float(gate_floor)
        if not 0.0 <= floor <= 1.0:
            raise ValueError("gate_floor must be in [0, 1]")
        multiplier = floor + (1.0 - floor) * alpha
        af = w.new_ones(G, T)
        af[:, :start] = torch.where(
            gate_valid[:, None],
            multiplier,
            w.new_ones(G, start),
        )
        w = _apply_prefix_gate(
            w,
            af,
            start,
            preserve_prefix_mass=preserve_prefix_mass,
        )
    if drop_pos:
        dp = [d for d in drop_pos if d < T]
        if dp:
            w[:, :, dp] = 0.0
    if scale_pos:
        sp = [d for d in scale_pos if d < T]
        if sp:
            w[:, :, sp] *= scale_factor
    O = torch.bmm(w, Vf)                                     # (G, 1, d)
    if normalize:
        O = O / w.sum(-1, keepdim=True).clamp_min(1e-8)
    return O


def causal_sv_readout_mlx_batched(Kf: torch.Tensor, Vf: torch.Tensor, Qf: torch.Tensor,
                                  C: float | Mapping[int, float], kpar: float,
                                  chunk: int, normalize: bool = True,
                                  gate: bool = True, fista_iters: int = 150,
                                  tol: float = 1e-3, state_sink: list | None = None,
                                  readout: str = "rbf", scaling: float | None = None,
                                  alpha_override: dict | None = None,
                                  drop_pos: list | None = None,
                                  scale_pos: list | None = None, scale_factor: float = 1.0,
                                  gate_sink: dict | None = None,
                                  preserve_prefix_mass: bool = False,
                                  solver_seed: int | None = None):
    """Chunk-frozen causal readout for a whole (batch*head) group at once.

    Kf, Vf, Qf: (G, T, *). At each chunk boundary the SVDD partition for ALL G
    problems is estimated by a single batched FISTA solve on the GPU (MLX); a
    ridge-stabilized KKT solve supplies differentiable coefficients conditional
    on those masks. Tests compare this approximation with the maintained
    chunk-frozen path at explicit tolerance. ``gate_sink`` (optional dict)
    receives the solved boundary gates so decode-time callers can reuse them.
    """
    G, T, _ = Kf.shape
    dv = Vf.shape[2]
    gates = compute_boundary_gates(
        Kf, C, kpar, chunk,
        gate=gate, fista_iters=fista_iters, tol=tol, state_sink=state_sink,
        alpha_override=alpha_override, drop_pos=drop_pos,
        solver_seed=solver_seed,
    )
    if gate_sink is not None:
        gate_sink.update(gates)

    # q.k softmax scale: default 1/sqrt(d); callers grafting onto a base model pass the
    # base layer's own scaling (e.g. Gemma3's query_pre_attn_scalar**-0.5) for fidelity.
    inv_scale = (1.0 / math.sqrt(Kf.shape[-1])) if scaling is None else float(scaling)

    def gate_alpha(start):
        """Differentiable prefix gate (G, start) + validity, or (None, None)."""
        entry = gates.get(start)
        if entry is None:
            return None, None
        return entry

    if readout == "softmax":
        # Hybrid: SV-gated long-range memory + LOCAL causal window. Query at global
        # position t attends (learned q.k softmax) to all of [0:t]; the chunk-frozen
        # prefix [0:start] is weighted by the solved gate alpha (reserve -> 0),
        # recent/local tokens [start:t] are always visible (alpha=1). The first chunk
        # is plain local causal attention (no prefix to gate).
        blocks = []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            qlen = end - start
            Qc = Qf[:, start:end]
            scores = Qc @ Kf[:, :end].transpose(-1, -2) * inv_scale     # (G, qlen, end)
            pos = torch.arange(end, device=Kf.device)
            qpos = start + torch.arange(qlen, device=Kf.device)
            cmask = pos[None, :] <= qpos[:, None]                       # (qlen, end) causal
            scores = scores.masked_fill(~cmask[None], float("-inf"))
            w = torch.softmax(scores, dim=-1)                          # (G, qlen, end)
            alpha, gate_valid = gate_alpha(start)
            if alpha is not None:
                af = w.new_ones(G, end)
                af[:, :start] = torch.where(gate_valid[:, None], alpha, w.new_ones(G, start))
                w = _apply_prefix_gate(
                    w,
                    af,
                    start,
                    preserve_prefix_mass=preserve_prefix_mass,
                )
            if drop_pos:                                       # forgotten/evicted keys contribute 0
                dp = [d for d in drop_pos if d < end]
                if dp:
                    w[:, :, dp] = 0.0
            if scale_pos:                                      # decay baseline: downweight, not evict
                sp = [d for d in scale_pos if d < end]
                if sp:
                    w[:, :, sp] *= scale_factor
            O = torch.bmm(w, Vf[:, :end])
            if normalize:
                O = O / w.sum(-1, keepdim=True).clamp_min(1e-8)
            blocks.append(O)
        return torch.cat(blocks, dim=1)

    blocks = [Qf.new_zeros(G, min(chunk, T), dv)]           # rbf: first chunk zeros
    for start in range(chunk, T, chunk):
        end = min(start + chunk, T)
        Vpref, Qc = Vf[:, :start], Qf[:, start:end]
        w = _batched_rbf(Qc, Kf[:, :start], kpar)            # (G, qlen, start)
        alpha, gate_valid = gate_alpha(start)
        if alpha is not None:
            wg = w * alpha.unsqueeze(1)                      # (G, qlen, start)
            w = torch.where(gate_valid[:, None, None], wg, w)  # empty-S -> ungated
        O = torch.bmm(w, Vpref)                              # (G, qlen, dv)
        if normalize:
            O = O / w.sum(-1, keepdim=True).clamp_min(1e-8)
        blocks.append(O)
    return torch.cat(blocks, dim=1)                          # (G, T, dv)


class CausalSVAttention(nn.Module):
    """Multi-head causal chunk-frozen SV-Attention (self-attention drop-in).

    forward(X): (B, T, d_model) -> (B, T, d_model). One independent one-class
    gate per (batch, head). The per-(batch, head) loop is the clarity-first v1;
    vectorizing it is the G1 throughput task.
    """

    def __init__(self, d_model: int, n_heads: int = 1, head_dim: int | None = None,
                 C: float = 0.25, kpar: float = 2.0, chunk: int = 16,
                 normalize: bool = True, exact_sequential: bool = False,
                 gate: bool = True, persistent: bool = False, solver: str = "exact",
                 fista_iters: int = 80, readout: str = "rbf",
                 preserve_prefix_mass: bool = False):
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim or d_model // n_heads)
        hd = self.head_dim * self.n_heads
        self.Wk = nn.Linear(d_model, hd, bias=False)
        self.Wq = nn.Linear(d_model, hd, bias=False)
        self.Wv = nn.Linear(d_model, hd, bias=False)
        self.Wo = nn.Linear(hd, d_model, bias=False)
        self.C, self.kpar, self.chunk, self.normalize = float(C), float(kpar), int(chunk), normalize
        self.exact_sequential = bool(exact_sequential)   # slow tokenwise reference
        self.gate = bool(gate)                           # gate=False -> uniform-RBF ablation
        self.persistent = bool(persistent)               # maintained rank-1 solver (G1)
        self.solver = str(solver)                        # "exact" (CPU active-set) | "mlx" (batched GPU)
        self.fista_iters = int(fista_iters)
        self.readout = str(readout)                      # "rbf" (distance) | "softmax" (learned q.k routing)
        self.preserve_prefix_mass = bool(preserve_prefix_mass)
        self.record_state = False                        # log mean active |S u E| per fwd
        self.state_log: list[float] = []

    def forward(self, X: torch.Tensor):
        B, T, _ = X.shape
        H, hd = self.n_heads, self.head_dim
        K = self.Wk(X).view(B, T, H, hd)
        Q = self.Wq(X).view(B, T, H, hd)
        V = self.Wv(X).view(B, T, H, hd)
        if self.solver == "mlx" and not self.exact_sequential:
            G = B * H
            Kf = K.permute(0, 2, 1, 3).reshape(G, T, hd)
            Vf = V.permute(0, 2, 1, 3).reshape(G, T, hd)
            Qf = Q.permute(0, 2, 1, 3).reshape(G, T, hd)
            sink = [] if self.record_state else None
            of = causal_sv_readout_mlx_batched(
                Kf, Vf, Qf, self.C, self.kpar, self.chunk, self.normalize,
                self.gate, self.fista_iters, state_sink=sink, readout=self.readout,
                preserve_prefix_mass=self.preserve_prefix_mass)
            if sink:
                self.state_log.append(sum(sink) / len(sink))
            out = of.reshape(B, H, T, hd).permute(0, 2, 1, 3)  # (B, T, H, hd)
            return self.Wo(out.reshape(B, T, H * hd))

        sink = [] if self.record_state else None
        rows = []
        for b in range(B):
            heads = []
            for h in range(H):
                if self.persistent and not self.exact_sequential:
                    o = causal_sv_readout_persistent(
                        K[b, :, h], V[b, :, h], Q[b, :, h], self.C, self.kpar,
                        self.chunk, self.normalize, self.gate, state_sink=sink)
                else:
                    o = causal_sv_readout(
                        K[b, :, h], V[b, :, h], Q[b, :, h], self.C, self.kpar,
                        self.chunk, self.normalize, self.exact_sequential, self.gate)
                heads.append(o)
            rows.append(torch.stack(heads, dim=1))       # (T, H, hd)
        if sink:
            self.state_log.append(sum(sink) / len(sink))
        out = torch.stack(rows, dim=0)                   # (B, T, H, hd)
        return self.Wo(out.reshape(B, T, H * hd))

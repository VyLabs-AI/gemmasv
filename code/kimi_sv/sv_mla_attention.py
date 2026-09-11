"""A certified support-vector gate in place of softmax on Kimi's global MLA layers.

This is the MLX analogue of ``gemma_sv.sv_global_attention``, and it exists to
test a property of the host architecture rather than to port code. Gemma gives
each head its own keys, so the gate must be fit once per (batch, head) group and
a record's deletion has to be reconciled across 32 partially disagreeing gates.
Kimi's multi-head latent attention compresses keys and values into one shared
``kv_lora_rank``-dimensional latent per token, which all heads then read. That
admits a single certified selection per *layer*:

``mode="latent"``
    One SVDD over the shared latent. One problem per layer, and the inert set is
    a single coherent decision.
``mode="per_head"``
    One SVDD per (batch, head) over materialized per-head keys, mirroring the
    Gemma graft. Present so the ablation can ask what, if anything, latent-space
    selection gives up.

Fidelity notes, following the companion paper:

* the gate is the one-class SVDD over the keys with box ``C = 1/(nu*n)``, and a
  reserve token (alpha = 0) contributes exactly zero to the readout;
* the readout is the hybrid used by the Gemma graft -- learned softmax weights
  with the chunk-frozen prefix multiplied by alpha -- not the capability paper's
  pure RBF average, because the pure gate plateaus as a language-model layer;
* the forward uses the single-precision FISTA solver. Exact removal is not done
  here; that is the float64 decrement path, as in ``gemma_sv.unlearn``.

The readout is computed in query blocks. A gated softmax cannot use MLX's fused
attention kernel, and materializing ``(heads, queries, keys)`` weights in one
piece would cost tens of gigabytes per layer at long context; blocking bounds it
and matches the granularity the chunk-frozen gate already has.

Graft *after* loading weights. Wrapping the base module nests its parameters
under ``self_attn.base.*``, which no longer matches the checkpoint's key names.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from svattn.mlx_svdd import svdd_fista_mlx

# Bandwidth is estimated from at most this many keys. The median over a
# subsample is stable enough that one extra token cannot move it beyond its own
# sampling noise, which is what makes freezing it at prefill defensible.
KPAR_SAMPLE = 256

# "shuffled" is the matched control for the question "is the gate selecting well,
# or merely intervening gently?" It solves the same latent SVDD and then permutes
# alpha along positions, so the multiset of gate weights is identical and only
# the assignment of weights to tokens is destroyed.
MODES = {"latent", "per_head", "shuffled"}


def median_bandwidth(keys: mx.array, sample: int = KPAR_SAMPLE) -> float:
    """Median pairwise key distance, the heuristic the Gemma graft calibrates to."""
    flat = keys.reshape(-1, keys.shape[-1]).astype(mx.float32)
    n = int(flat.shape[0])
    if n > sample:
        idx = mx.array(
            [round(i * (n - 1) / (sample - 1)) for i in range(sample)], dtype=mx.int32
        )
        flat = flat[idx]
    sq = mx.sum(flat * flat, axis=-1)
    d2 = mx.maximum(sq[:, None] + sq[None, :] - 2.0 * (flat @ flat.T), 0.0)
    n = int(flat.shape[0])
    iu = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not iu:
        return 1.0
    rows = mx.array([i for i, _ in iu], dtype=mx.int32)
    cols = mx.array([j for _, j in iu], dtype=mx.int32)
    dists = mx.sqrt(d2[rows, cols])
    med = float(mx.median(dists).item())
    return med if med > 1e-6 else 1.0


def _keep_vector(drop_pos: Optional[List[int]], length: int) -> Optional[mx.array]:
    """A ``(length,)`` 0/1 vector zeroing dropped positions, or ``None``."""
    dropped = [d for d in (drop_pos or ()) if 0 <= d < length]
    if not dropped:
        return None
    positions = mx.arange(length)
    targets = mx.array(dropped, dtype=positions.dtype)
    hit = mx.sum((positions[None, :] == targets[:, None]).astype(mx.float32), axis=0)
    return 1.0 - mx.minimum(hit, 1.0)


def rbf_gram(a: mx.array, b: mx.array, kpar: float) -> mx.array:
    """Batched ``exp(-||a-b||^2 / kpar^2)`` for ``(G, m, d)`` and ``(G, n, d)``."""
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    a2 = mx.sum(a * a, axis=-1)[..., :, None]
    b2 = mx.sum(b * b, axis=-1)[..., None, :]
    d2 = mx.maximum(a2 + b2 - 2.0 * (a @ b.swapaxes(-1, -2)), 0.0)
    return mx.exp(-d2 / (kpar * kpar))


class SVMLAAttention(nn.Module):
    """Drop-in for ``KimiMLAAttention`` whose prefix readout is SV-gated."""

    def __init__(
        self,
        base: nn.Module,
        *,
        mode: str = "latent",
        nu: float = 0.3,
        C: Optional[float] = None,
        kpar: Optional[float] = None,
        chunk: int = 128,
        gate: bool = True,
        fista_iters: int = 80,
        normalize: bool = True,
        preserve_prefix_mass: bool = False,
        collect_stats: bool = False,
        solver_seed: int = 0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
        self.base = base
        self.mode = mode
        self.nu = float(nu)
        self.C = None if C is None else float(C)
        self.kpar = None if kpar is None else float(kpar)
        self.chunk = int(chunk)
        self.gate = bool(gate)
        self.fista_iters = int(fista_iters)
        self.normalize = bool(normalize)
        # The SVDD dual sums to one, so as the prefix grows every gated weight
        # shrinks as ~1/(nu*n) and the prefix's total influence fades relative
        # to the ungated local window -- measured as a recall collapse by ~768
        # tokens, and not fixable by the box alone (a larger C trades coverage
        # for weight). Preserving the prefix's pre-gate mass share makes the
        # gate a pure redistribution *within* the prefix: reserves still carry
        # exactly zero (the certificate is scale-invariant), but the prefix as
        # a whole keeps the influence plain softmax gave it.
        self.preserve_prefix_mass = bool(preserve_prefix_mass)
        self.collect_stats = bool(collect_stats)

        self.is_linear = False
        # The batched FISTA solver seeds its power iteration for the Lipschitz
        # estimate with mx.random.normal, so two solves over identical keys can
        # disagree at ~1e-7 and blur any claim about exact deletion. Reseeding
        # before each solve makes the gate reproducible. The shared solver in
        # svattn/ is left untouched because the companion paper depends on it.
        self.solver_seed = int(solver_seed)
        # Positions excluded from the gate fit and forced to zero readout weight:
        # the eviction / forgetting path, equivalent to a refit without them at
        # solver precision.
        self.drop_pos: Optional[List[int]] = None
        self.stats: Dict[str, Any] = {}
        self.last_alpha: Optional[mx.array] = None
        self.last_boundary: Optional[int] = None
        # Accumulated raw softmax mass per absolute key position -- the
        # heavy-hitter (H2O-style) eviction score. The eviction evaluation turns
        # this on for prefill only, with ``gate=False`` so the accumulated weights
        # are plain attention; probes leave it off so measurement does not
        # contaminate the scores.
        self.h2o_accumulate = False
        self.h2o_scores: Optional[mx.array] = None

    # -- gate ------------------------------------------------------------

    def _box(self, n: int) -> float:
        if self.C is not None:
            return self.C
        return 1.0 / max(self.nu * n, 1e-6)

    def _gate_keys(self, kv_latent: mx.array) -> mx.array:
        """Keys the gate is fit over, shaped ``(G, S, d)``."""
        if self.mode in {"latent", "shuffled"}:
            return kv_latent[:, 0].astype(mx.float32)
        per_head = self.base.embed_q(kv_latent, transpose=False)
        b, h, s, d = per_head.shape
        return per_head.reshape(b * h, s, d).astype(mx.float32)

    def _solve_boundaries(
        self, keys: mx.array, boundaries: List[int]
    ) -> Dict[int, mx.array]:
        """Chunk-boundary SVDD gates: ``{boundary: alpha (G, boundary)}``.

        Each boundary is solved on its own, because the box ``C = 1/(nu*n)``
        depends on the number of points in that problem and the batched solver
        takes a single scalar box. Solving separately also keeps peak memory at
        the largest single gram rather than every boundary padded to it, which is
        what makes the per-head mode affordable.
        """
        if not boundaries:
            return {}

        alphas: Dict[int, mx.array] = {}
        for s in boundaries:
            gram = rbf_gram(keys[:, :s], keys[:, :s], self._kpar_value)
            mask = mx.ones((int(keys.shape[0]), s), dtype=mx.float32)
            keep = _keep_vector(self.drop_pos, s)
            if keep is not None:
                mask = mask * keep[None, :]
            mx.random.seed(self.solver_seed)
            alpha = svdd_fista_mlx(
                gram, self._box(s), iters=self.fista_iters, mask=mask
            )
            if self.mode == "shuffled":
                # Same weights, deliberately wrong targets.
                mx.random.seed(self.solver_seed + s)
                order = mx.argsort(mx.random.uniform(shape=(alpha.shape[0], s)), axis=1)
                alpha = mx.take_along_axis(alpha, order, axis=1)
            alphas[s] = alpha
        return alphas

    def _boundaries(self, total: int) -> List[int]:
        """Chunk boundaries whose prefix is large enough to gate."""
        if not self.gate:
            return []
        if self.C is None:
            # With C = 1/(nu*n) the box always admits unit mass for nu <= 1
            # (n*C = 1/nu), so only a meaningful number of points is required.
            threshold = 3
        else:
            # A fixed box needs at least 1/C points to carry unit mass.
            threshold = math.ceil(1.0 / self.C) + 2
        return [s for s in range(self.chunk, total, self.chunk) if s >= threshold]

    def _record_stats(self, alphas: Dict[int, mx.array], total: int) -> None:
        # The last boundary's gate is kept so callers can read which positions the
        # active-set partition certified inert.
        self.last_boundary = max(alphas) if alphas else None
        self.last_alpha = alphas[self.last_boundary] if alphas else None

        if not (self.collect_stats and alphas):
            return
        last = self.last_boundary
        alpha = alphas[last]
        box = self._box(last)
        inert = mx.sum(alpha <= 1e-8, axis=-1)
        self.stats = {
            "mode": self.mode,
            "problems": int(alpha.shape[0]),
            "boundary": last,
            "context_tokens": total,
            "support_mean": float(mx.mean(mx.sum(alpha > 1e-8, axis=-1)).item()),
            "inert_mean": float(mx.mean(inert).item()),
            "inert_fraction": float(mx.mean(inert).item()) / max(last, 1),
            "at_box_mean": float(mx.mean(mx.sum(alpha >= box - 1e-8, axis=-1)).item()),
        }

    # -- forward ---------------------------------------------------------

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        b = self.base
        batch, length, _ = x.shape

        q = b.q_proj(x).reshape(batch, length, b.num_heads, b.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [b.qk_nope_head_dim], axis=-1)

        compressed = b.kv_a_proj_with_mqa(x)
        compressed, k_pe = mx.split(compressed, [b.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(batch, length, 1, b.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(b.kv_a_layernorm(compressed), axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        total = int(kv_latent.shape[2])
        offset = total - length

        gate_keys = self._gate_keys(kv_latent)
        if self.kpar is None:
            self._kpar_value = median_bandwidth(gate_keys)
            self.kpar = self._kpar_value
        else:
            self._kpar_value = self.kpar

        # A query at global position p reads the gate frozen at the boundary at
        # or before p, so only boundaries some query in this forward actually
        # uses need solving -- for a short probe over a long context that is one
        # solve, not one per chunk of history.
        needed = {((offset + j) // self.chunk) * self.chunk for j in range(length)}
        alphas = self._solve_boundaries(
            gate_keys, [s for s in self._boundaries(total) if s in needed]
        )
        self._record_stats(alphas, total)
        per_head_gate = self.mode == "per_head"

        if length == 1:
            queries = b.embed_q(q_nope)
            keys = values = kv_latent
        else:
            queries = q_nope
            keys = b.embed_q(kv_latent, transpose=False)
            values = b.unembed_out(kv_latent)

        outputs = []
        retained = 0.0
        counted = 0
        for start in range(0, length, self.chunk):
            end = min(start + self.chunk, length)
            visible = offset + end
            block_q = queries[:, :, start:end]

            scores = (block_q * b.scale) @ keys[..., :visible, :].swapaxes(-1, -2)
            scores = scores + (q_pe[:, :, start:end] * b.scale) @ k_pe[
                ..., :visible, :
            ].swapaxes(-1, -2)
            if mask is not None:
                block_mask = mask[start:end, :visible]
                scores = mx.where(
                    block_mask,
                    scores,
                    mx.array(mx.finfo(scores.dtype).min, scores.dtype),
                )
            weights = mx.softmax(scores.astype(mx.float32), axis=-1)

            if self.h2o_accumulate:
                contrib = mx.sum(weights, axis=(0, 1, 2))
                n = int(contrib.shape[0])
                if self.h2o_scores is None:
                    self.h2o_scores = mx.zeros((n,), dtype=mx.float32)
                m = int(self.h2o_scores.shape[0])
                if m < n:
                    self.h2o_scores = mx.concatenate(
                        [self.h2o_scores, mx.zeros((n - m,), dtype=mx.float32)]
                    )
                elif m > n:
                    contrib = mx.concatenate(
                        [contrib, mx.zeros((m - n,), dtype=mx.float32)]
                    )
                self.h2o_scores = self.h2o_scores + contrib
                mx.eval(self.h2o_scores)

            factors = self._block_factors(
                alphas, offset + start, end - start, visible, batch, b.num_heads,
                per_head_gate, weights.dtype,
            )
            if factors is not None:
                if self.preserve_prefix_mass:
                    # Rescale the gated prefix back to its pre-gate mass share,
                    # per query row. Zeros stay zero, so the certificate holds.
                    local = (factors == 1.0).astype(mx.float32)
                    before = mx.sum(weights * (1.0 - local), axis=-1, keepdims=True)
                    gated = weights * factors
                    after = mx.sum(gated * (1.0 - local), axis=-1, keepdims=True)
                    scale = before / mx.maximum(after, 1e-12)
                    weights = gated * (local + (1.0 - local) * scale)
                else:
                    weights = weights * factors
                if self.collect_stats:
                    # Softmax mass surviving the gate, before renormalization.
                    # This separates "the gate zeroes many positions" from "the
                    # gate removes much of what the model was attending to".
                    retained += float(mx.sum(mx.sum(weights, axis=-1)).item())
                    counted += int(weights.size // weights.shape[-1])

            out = weights.astype(values.dtype) @ values[..., :visible, :]
            if self.normalize:
                total_weight = mx.sum(weights, axis=-1, keepdims=True)
                out = out / mx.maximum(total_weight, 1e-8).astype(values.dtype)
            outputs.append(out)

        if self.collect_stats and counted:
            self.stats["mass_retained_mean"] = retained / counted

        output = mx.concatenate(outputs, axis=2)
        if length == 1:
            output = b.unembed_out(output)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return b.o_proj(output)

    def _block_factors(
        self,
        alphas: Dict[int, mx.array],
        first_position: int,
        n_queries: int,
        visible: int,
        batch: int,
        heads: int,
        per_head_gate: bool,
        dtype: Any,
    ) -> Optional[mx.array]:
        """Per-query readout weights for the frozen prefix, or ``None`` if ungated.

        Query at global position ``p`` uses the gate solved at the last chunk
        boundary at or before ``p``; keys after that boundary are local and stay
        at weight one.
        """
        if not alphas and not self.drop_pos:
            return None

        problems = batch * heads if per_head_gate else batch
        keep = _keep_vector(self.drop_pos, visible)

        # Consecutive queries in a block share a boundary unless the block
        # straddles one, so build one row per run rather than one per query.
        runs: List[Tuple[int, int]] = []
        for i in range(n_queries):
            boundary = ((first_position + i) // self.chunk) * self.chunk
            if runs and runs[-1][0] == boundary:
                runs[-1] = (boundary, runs[-1][1] + 1)
            else:
                runs.append((boundary, 1))

        pieces = []
        for boundary, count in runs:
            row = mx.ones((problems, visible), dtype=mx.float32)
            alpha = alphas.get(boundary)
            if alpha is not None:
                row = mx.concatenate(
                    [alpha.astype(mx.float32), row[:, boundary:]], axis=1
                )
            if keep is not None:
                row = row * keep[None, :]
            pieces.append(mx.broadcast_to(row[:, None, :], (problems, count, visible)))

        factors = mx.concatenate(pieces, axis=1)
        if per_head_gate:
            factors = factors.reshape(batch, heads, n_queries, visible)
        else:
            factors = mx.expand_dims(factors, axis=1)
        return factors.astype(dtype)

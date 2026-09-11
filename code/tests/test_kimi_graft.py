"""The SV gate grafted onto Kimi's global MLA layers.

Checks the properties the gate is supposed to have -- it only touches global
layers, it is reproducible, its certified-inert set can be evicted without
changing the readout, and replay deletion stays exact with the gate in place --
on the miniature random model, where a bug is cheap to find.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from kimi_sv.graft import (
    gate_stats,
    graft_sv_into_kimi,
    grafted_layers,
    set_drop_positions,
)
from kimi_sv.records import Record, RecordMemory, max_abs_diff, state_max_abs_diff
from kimi_sv.state import classify_layers
from kimi_sv.sv_mla_attention import SVMLAAttention
from kimi_sv.tiny import build_tiny, token_block

MODES = ["latent", "per_head"]
# FISTA is an approximate solver, so the certificate holds to its tolerance; the
# exact guarantee is the float64 decrement path, not this forward.
SOLVER_TOL = 1e-4


def _grafted(mode: str, **kwargs):
    model, _ = build_tiny(seed=0)
    replaced = graft_sv_into_kimi(model, mode=mode, chunk=32, collect_stats=True, **kwargs)
    return model, replaced


def _records(n: int = 3, size: int = 40):
    return [
        Record(key=f"r{i}", tokens=token_block(size, start=100 * (i + 1)))
        for i in range(n)
    ]


@pytest.mark.parametrize("mode", MODES)
def test_graft_replaces_only_global_layers(mode):
    model, replaced = _grafted(mode)
    kda, global_layers = classify_layers(model)

    assert replaced == global_layers
    assert grafted_layers(model) == global_layers
    for i in kda:
        assert not isinstance(model.layers[i].self_attn, SVMLAAttention)


@pytest.mark.parametrize("mode", MODES)
def test_forward_is_finite_and_reproducible(mode):
    model, _ = _grafted(mode)
    tokens = token_block(96)

    first = model(tokens)
    second = model(tokens)
    mx.eval(first, second)

    assert bool(mx.all(mx.isfinite(first)).item())
    # Reseeding the solver's power iteration makes the gate reproducible; without
    # it, repeated forwards disagree at ~1e-7 and blur the deletion claim.
    assert max_abs_diff(first, second) == 0.0


def test_latent_mode_solves_one_problem_per_layer():
    latent, layers = _grafted("latent")
    latent(token_block(96))
    per_head, _ = _grafted("per_head")
    per_head(token_block(96))

    n_heads = latent.layers[layers[0]].self_attn.base.num_heads
    assert gate_stats(latent)[layers[0]]["problems"] == 1
    assert gate_stats(per_head)[layers[0]]["problems"] == n_heads


@pytest.mark.parametrize("mode", MODES)
def test_gate_certifies_some_tokens_inert(mode):
    model, layers = _grafted(mode)
    model(token_block(96))

    stats = gate_stats(model)[layers[0]]
    assert 0.0 < stats["inert_fraction"] <= 1.0
    assert stats["support_mean"] + stats["inert_mean"] == pytest.approx(
        stats["boundary"], rel=1e-6
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("preserve", [False, True])
def test_evicting_certified_inert_tokens_preserves_the_readout(mode, preserve):
    """The certificate: a reserve token carries no weight, so evicting it is free.

    The guarantee is scoped by the chunk-frozen schedule. A token is *local* --
    ungated, full weight -- until the next chunk boundary, and only the gate
    solved at that boundary certifies it inert. So the invariant holds for
    queries at or after the boundary, not for earlier ones, which legitimately
    saw the token in their local window.

    Reserve points also do not constrain the SVDD solution, so dropping them
    leaves the retained alphas alone; that convergence is why the comparison is
    to solver tolerance rather than exact. The mass-preserving readout rescales
    the gated prefix by a factor that is invariant to dropping zero-weight
    positions, so the certificate must survive it unchanged.
    """
    model, layers = _grafted(mode, fista_iters=600, preserve_prefix_mass=preserve)
    tokens = token_block(96)
    before = model(tokens)
    mx.eval(before)

    gate = model.layers[layers[0]].self_attn
    alpha, boundary = gate.last_alpha, gate.last_boundary
    assert alpha is not None

    # Positions inert across every problem in this layer.
    inert = mx.all(alpha <= 1e-8, axis=0)
    positions = [i for i in range(int(boundary)) if bool(inert[i].item())]
    if not positions:
        pytest.skip("no position was inert across all problems")

    set_drop_positions(model, positions)
    after = model(tokens)
    mx.eval(after)

    gated = slice(int(boundary), None)
    assert max_abs_diff(before[:, gated], after[:, gated]) < SOLVER_TOL
    # And the eviction is not a no-op overall: earlier queries held the token in
    # their local window, so their readout does move.
    assert max_abs_diff(before, after) > SOLVER_TOL


@pytest.mark.parametrize("mode", MODES)
def test_replay_deletion_stays_exact_with_the_gate(mode):
    records = _records()
    probe = token_block(6, start=900)

    model, _ = _grafted(mode)
    memory = RecordMemory(model)
    memory.ingest_all(records)
    memory.delete("r1")

    ref_model, _ = _grafted(mode)
    reference = RecordMemory(ref_model)
    reference.ingest_all([r for r in records if r.key != "r1"])

    assert max_abs_diff(memory.probe(probe), reference.probe(probe)) == 0.0
    assert state_max_abs_diff(memory, reference) == 0.0


def test_gate_can_be_disabled():
    model, layers = _grafted("latent", gate=False)
    model(token_block(96))
    assert gate_stats(model) == {}


def test_h2o_scores_accumulate_plain_softmax_mass():
    """Heavy-hitter scores: per-position softmax mass, totalling heads x queries."""
    model, layers = _grafted("latent", gate=False)
    attn = model.layers[layers[0]].self_attn
    attn.h2o_accumulate = True

    n = 96
    model(token_block(n))

    scores = attn.h2o_scores
    assert scores is not None and int(scores.shape[0]) == n
    assert bool(mx.all(scores >= 0.0).item())
    # Every query's softmax row sums to one, so the accumulated mass must equal
    # batch * heads * queries.
    expected = 1 * attn.base.num_heads * n
    assert float(mx.sum(scores).item()) == pytest.approx(expected, rel=1e-4)


def test_drop_positions_accept_per_layer_dict():
    model, layers = _grafted("latent")
    attn = model.layers[layers[0]].self_attn

    set_drop_positions(model, {layers[0]: [0, 1]})
    assert attn.drop_pos == [0, 1]

    # A dict without an entry for a grafted layer clears that layer.
    set_drop_positions(model, {})
    assert attn.drop_pos is None

    set_drop_positions(model, [2, 3])
    assert attn.drop_pos == [2, 3]
    set_drop_positions(model, None)
    assert attn.drop_pos is None


def test_unknown_mode_rejected():
    model, _ = build_tiny(seed=0)
    with pytest.raises(ValueError, match="mode must be"):
        graft_sv_into_kimi(model, mode="nonsense")

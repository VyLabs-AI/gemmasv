"""The separability harness's write and transport contracts.

Locks in the algebra behind ``kimi_sv.eval_separability``: a record's state
contribution is suffix-independent under the separable rule, decay-correctable
under the decay-only rule, and full-transition transportable under KDA when
future kernel inputs are frozen. A changed suffix adds transition and write
forcing that replay must recompute.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from kimi_sv.eval_separability import (
    decompose_kda_suffix_difference,
    rel_diff,
    run_kda,
    run_rule,
    suffix_decay,
    take_tokens,
    transported_record_receipt,
)

B, H, DK, DV = 1, 4, 16, 16
N_PREFIX, N_VICTIM, N_SUFFIX = 20, 10, 30
T = N_PREFIX + N_VICTIM + N_SUFFIX


def _inputs(seed: int) -> dict:
    mx.random.seed(seed)
    return {
        "q": mx.random.normal((B, T, H, DK)) * 0.3,
        "k": mx.random.normal((B, T, H, DK)) * 0.3,
        "v": mx.random.normal((B, T, H, DV)) * 0.3,
        "g": 0.98 + 0.019 * mx.random.uniform(shape=(B, T, H, DK)),
        "beta": mx.sigmoid(mx.random.normal((B, T, H))),
    }


@pytest.fixture(scope="module")
def table():
    cap_a, cap_b = _inputs(1), _inputs(2)
    for name in cap_a:  # same prefix and victim inputs, different suffixes
        cap_b[name][:, : N_PREFIX + N_VICTIM] = cap_a[name][:, : N_PREFIX + N_VICTIM]

    with_idx = list(range(T))
    prefix_idx = list(range(N_PREFIX))
    victim_idx = list(range(N_PREFIX, N_PREFIX + N_VICTIM))
    skip_idx = list(range(N_PREFIX)) + list(range(N_PREFIX + N_VICTIM, T))
    suffix_idx = list(range(N_PREFIX + N_VICTIM, T))

    d_a = suffix_decay(cap_a, suffix_idx)[..., None, :]
    d_b = suffix_decay(cap_b, suffix_idx)[..., None, :]

    out = {}
    for rule in ("separable", "decay", "kda"):
        run = run_kda if rule == "kda" else (lambda c, r=rule: run_rule(c, r))
        ds_a = run(take_tokens(cap_a, with_idx)) - run(take_tokens(cap_a, skip_idx))
        ds_b = run(take_tokens(cap_b, with_idx)) - run(take_tokens(cap_b, skip_idx))
        out[rule] = {
            "raw": rel_diff(ds_a, ds_b),
            "corrected": rel_diff(ds_a * d_b, ds_b * d_a),
            "norm": max(
                float(mx.sqrt(mx.sum(ds_a * ds_a)).item()),
                float(mx.sqrt(mx.sum(ds_b * ds_b)).item()),
            ),
        }
        if rule == "kda":
            transported_a = transported_record_receipt(
                cap_a, prefix_idx, victim_idx, suffix_idx
            )
            transported_b = transported_record_receipt(
                cap_b, prefix_idx, victim_idx, suffix_idx
            )
            out[rule]["transport_error"] = max(
                rel_diff(ds_a, transported_a),
                rel_diff(ds_b, transported_b),
            )
    return out


def test_contributions_are_not_degenerate(table):
    for rule in ("separable", "decay", "kda"):
        assert table[rule]["norm"] > 1e-3


def test_separable_rule_is_suffix_independent(table):
    assert table["separable"]["raw"] < 1e-4


def test_decay_rule_is_suffix_dependent_but_ledger_correctable(table):
    assert table["decay"]["raw"] > 1e-3
    assert table["decay"]["corrected"] < 1e-5


def test_delta_rule_is_not_correctable_by_static_or_decay_ledger(table):
    assert table["kda"]["raw"] > 0.05
    assert table["kda"]["corrected"] > 0.05
    assert table["kda"]["corrected"] > 100 * table["decay"]["corrected"]


def test_delta_rule_receipt_is_exactly_full_transition_transportable(table):
    assert table["kda"]["transport_error"] < 1e-5


def test_changed_suffix_forcing_decomposition_is_exact():
    present, omitted = _inputs(7), _inputs(8)
    # The prefix is identical; the present branch additionally ingests the
    # victim before the two branches receive different suffix transitions.
    for name in present:
        omitted[name][:, :N_PREFIX] = present[name][:, :N_PREFIX]

    prefix_idx = list(range(N_PREFIX))
    victim_idx = list(range(N_PREFIX, N_PREFIX + N_VICTIM))
    suffix_idx = list(range(N_PREFIX + N_VICTIM, T))
    omitted_suffix_idx = list(range(N_PREFIX, N_PREFIX + N_SUFFIX))

    prefix_state = run_kda(take_tokens(present, prefix_idx))
    initial_present = run_kda(
        take_tokens(present, victim_idx), state=prefix_state
    )
    initial_omitted = prefix_state
    present_suffix = take_tokens(present, suffix_idx)
    omitted_suffix = take_tokens(omitted, omitted_suffix_idx)
    components = decompose_kda_suffix_difference(
        initial_present,
        initial_omitted,
        present_suffix,
        omitted_suffix,
    )

    final_present = run_kda(present_suffix, state=initial_present)
    final_omitted = run_kda(omitted_suffix, state=initial_omitted)
    direct = final_present - final_omitted
    assert rel_diff(components["total"], direct) < 1e-5
    forcing = (
        components["transition_forcing"]
        + components["write_forcing"]
    )
    assert rel_diff(forcing, mx.zeros_like(forcing)) > 0.1

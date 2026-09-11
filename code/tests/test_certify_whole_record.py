from types import SimpleNamespace

import numpy as np

from gemma_sv.certify_whole_record import certify_probe


class _Runtime:
    def __init__(self):
        self.calls = []

    def target_ids(self, target, prompt):
        assert target == "secret"
        assert prompt == "prompt"
        return [7]

    def score_persistent(self, state, prompt, target_ids):
        self.calls.append((state.name, prompt, target_ids))
        shift = {"exact": 0.0, "refit": 1e-9, "decay": 0.1}[state.name]
        logits = np.log(np.array([0.7 - shift, 0.3 + shift]))
        return {
            "first_log_probs": logits,
            "geometric_mean_probability": 0.5,
        }

    def top_tokens(self, distribution):
        return [{"token": "x", "probability": float(np.exp(distribution[0]))}]


def test_certify_probe_uses_sealed_persistent_states():
    runtime = _Runtime()
    states = {
        name: SimpleNamespace(name=name)
        for name in ("exact", "refit", "decay")
    }
    overrides = {
        "fixed_c_feasible": True,
        "n_fallback": 2,
        "n_solves": 12,
        "max_functional_deviation": 1e-8,
        "max_candidate_deviation": 2e-8,
    }

    result = certify_probe(runtime, states, overrides, "prompt", "secret")

    assert [call[0] for call in runtime.calls] == ["exact", "refit", "decay"]
    assert result["fixed_c_feasible"] is True
    assert result["decrement_fallbacks"] == 2
    assert result["head_gates"] == 12
    assert result["kl_exact_vs_refit_nats"] < 1e-12
    assert result["kl_decay_vs_refit_nats"] > 0

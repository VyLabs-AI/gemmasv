import pytest

from gemma_sv.publish_arxiv_v2_claims import (
    certificate_summary,
    multiseed_scale_result,
    scale_result,
)


def test_scale_result_reports_matched_percent_cost():
    recovered = {
        "model": "model",
        "ppl_final": 11.0,
        "config": {"seed": 0, "eval_blocks": 400, "rank": 8},
    }
    control = {"ppl_control": 10.0}
    result = scale_result(recovered, control)
    assert result["utility_cost_percent"] == pytest.approx(10.0)
    assert result["evaluation_blocks"] == 400


def test_multiseed_scale_result_accepts_five_complete_pairs():
    payload = {
        "schema": "gemma-sv-4b-multiseed-v1",
        "status": "complete",
        "n_seeds": 5,
        "model": "google/gemma-3-4b-pt",
        "model_revision": "revision",
        "seeds": [0, 1, 2, 3, 4],
        "seed_rows": [{"seed": seed} for seed in range(5)],
        "recovered_ppl": {"mean": 15.0},
        "control_ppl": {"mean": 14.0},
        "utility_cost_percent": {"mean": 7.1},
        "inference_unit": "matched training seed",
    }

    result = multiseed_scale_result(payload)

    assert result["n_seeds"] == 5
    assert result["seeds"] == [0, 1, 2, 3, 4]


def test_multiseed_scale_result_rejects_duplicate_seed_ids():
    payload = {
        "schema": "gemma-sv-4b-multiseed-v1",
        "status": "complete",
        "n_seeds": 3,
        "model": "google/gemma-3-4b-pt",
        "model_revision": "revision",
        "seeds": [0, 1, 1],
        "seed_rows": [{"seed": seed} for seed in [0, 1, 1]],
        "recovered_ppl": {},
        "control_ppl": {},
        "utility_cost_percent": {},
        "inference_unit": "matched training seed",
    }

    with pytest.raises(ValueError, match="incomplete or duplicated"):
        multiseed_scale_result(payload)


def test_certificate_summary_removes_tokens_and_record_labels():
    payload = {
        "records": [
            {
                "record_id": "private-label",
                "probe_certificates": {
                    "probe": {
                        "kl_exact_vs_refit_nats": 1e-12,
                        "decrement_fallbacks": 2,
                        "head_gates": 10,
                        "top_tokens": [{"token": "secret"}],
                    }
                },
            }
        ]
    }
    result = certificate_summary(payload)
    assert result == {
        "probe_count": 1,
        "maximum_exact_refit_kl_nats": 1e-12,
        "mean_exact_refit_kl_nats": 1e-12,
        "decrement_fallbacks": 2,
        "head_gates": 10,
    }

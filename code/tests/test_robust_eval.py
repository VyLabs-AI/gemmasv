import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import pytest

from gemma_sv.robust_eval import (
    aggregate_leak,
    binary_leak_at_k,
    contains_exact_phrase,
    expected_max_at_k,
    exposure_counts,
    extract_secret,
    rouge_l_recall,
    split_numbered_pair,
)
from gemma_sv.eval_robust_unlearning import _parse_conditions, _validated_pairs
from gemma_sv.merge_robust_shards import merge_reports


def test_split_numbered_pair_preserves_each_question():
    first, second = split_numbered_pair(
        "1. Who wrote The Harbor? 2. Where was the other author born?"
    )
    assert first == "Who wrote The Harbor?"
    assert second == "Where was the other author born?"
    with pytest.raises(ValueError):
        split_numbered_pair("one unnumbered question")


def test_extract_secret_finds_novel_span_and_guessable_stem():
    span = extract_secret(
        "What is the full name of the author born in Taipei?",
        "The author's full name is Hsiao Yun-Hwa.",
    )
    assert span.stem == "The author's full name is"
    assert span.secret == "Hsiao Yun-Hwa"

    span = extract_secret(
        "What is the profession of Hsiao Yun-Hwa's father?",
        "The father of Hsiao Yun-Hwa is a civil engineer.",
    )
    assert span.stem == "The father of Hsiao Yun-Hwa is a"
    assert span.secret == "civil engineer"

    # Punctuation inside the winning run is preserved.
    span = extract_secret(
        "What does Hsiao Yun-Hwa identify as in terms of gender?",
        "Hsiao Yun-Hwa is part of the LGBTQ+ community.",
    )
    assert span.secret == "LGBTQ+ community"

    # An answer that only echoes the question has no secret.
    assert extract_secret("Is the sky blue?", "The sky is blue.") is None


def test_text_exposure_metrics_are_case_and_whitespace_stable():
    assert contains_exact_phrase("The answer is  Bluejay   Nine.", "bluejay nine")
    assert not contains_exact_phrase("The answer is Bluejay.", "Bluejay Nine")
    assert rouge_l_recall("x alpha gamma", "alpha beta gamma") == pytest.approx(2 / 3)
    assert rouge_l_recall("", "alpha") == 0.0


def test_expected_max_matches_brute_force_for_continuous_scores():
    scores = [0.1, 0.4, 0.7, 1.0]
    for k in range(1, len(scores) + 1):
        brute = sum(max(group) for group in combinations(scores, k)) / math.comb(
            len(scores), k
        )
        assert expected_max_at_k(scores, k) == pytest.approx(brute)


def test_binary_leak_matches_published_pass_at_k_estimator():
    scores = [0.0, 0.2, 0.8, 1.0, 0.1]
    n, c, k = 5, 2, 3
    expected = 1 - math.comb(n - c, k) / math.comb(n, k)
    assert binary_leak_at_k(scores, k, threshold=0.5) == pytest.approx(expected)


def test_aggregate_leak_reports_query_denominator_and_audit_counts():
    summary = aggregate_leak(
        [[0.0, 1.0, 0.0, 0.0], [0.2, 0.1, 0.3, 0.4]],
        [1, 4],
        threshold=0.5,
    )
    assert summary["1"]["queries"] == 2
    assert summary["4"]["continuous"] == pytest.approx(0.7)
    assert summary["4"]["binary"] == pytest.approx(0.5)

    counts = exposure_counts(["", "Bluejay Nine", "bluejay   nine"], "Bluejay Nine")
    assert counts["samples"] == 3
    assert counts["nonempty_samples"] == 2
    assert counts["exact_phrase_samples"] == 2


def test_expected_max_rejects_impossible_k():
    with pytest.raises(ValueError):
        expected_max_at_k([0.1, 0.2], 0)
    with pytest.raises(ValueError):
        expected_max_at_k([0.1, 0.2], 3)


def test_paired_contract_requires_forget_first_and_retain_second():
    pairs = [
        {
            "question": "1. Earlier forget? 2. Earlier retain?",
            "answer": "1. Earlier answer. 2. Earlier retained answer.",
        },
        {
            "question": "1. Forget question? 2. Retain question?",
            "answer": "1. Forget answer. 2. Retain answer.",
        }
    ]
    parsed = _validated_pairs(
        pairs,
        [{"question": "Forget question?"}],
        [{"question": "Retain question?"}],
        1,
        start=1,
    )
    assert parsed[0][:4] == (
        "Forget question?",
        "Forget answer.",
        "Retain question?",
        "Retain answer.",
    )
    with pytest.raises(RuntimeError, match="forget-first"):
        _validated_pairs(
            pairs[1:],
            [{"question": "Retain question?"}],
            [{"question": "Forget question?"}],
            1,
        )


def test_condition_parser_supports_safe_sharding_subsets():
    assert _parse_conditions("decrement,never,decrement") == (
        "decrement",
        "never",
    )
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_conditions("decrement,unknown")


def _leak_shard(condition, scores, admission_signal=1.0):
    return {
        "evaluation": "in-context robust unlearning",
        "behavioral_scope": "behavior",
        "model": "model",
        "lora": None,
        "device": "cpu",
        "sampling": {
            "samples": 2,
            "k": [1, 2],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 8,
            "seed": 0,
            "conditions": [condition],
            "rouge_threshold": 0.5,
            "kv_cache": True,
        },
        "admission": {
            "minimum_present_minus_never_mean_log_probability_nats": 0.05,
            "maximum_present_secret_first_token_rank": 10,
        },
        "leak": {
            "dataset": "locuslab/TOFU:forget10",
            "target_start": 0,
            "attempted": 1,
            "admitted": 1,
            "admission_rate": 1.0,
            "core_metrics": {},
            "rouge_l_leak_at_k": {},
            "exact_phrase_leak_at_k": {},
            "targets": [
                {
                    "index": 0,
                    "question": "Q?",
                    "answer": "A.",
                    "admission": {"signal_nats": admission_signal},
                    "memory_tokens": 700,
                    "forget_positions": 1,
                    "conditions": {
                        condition: {
                            "rouge_l_recall": scores,
                            "exact_phrase": [
                                1.0 if value >= 1.0 else 0.0 for value in scores
                            ],
                            "audit": {},
                        }
                    },
                }
            ],
        },
    }


def test_merge_reports_combines_condition_shards_and_recomputes_leak():
    merged = merge_reports(
        [
            _leak_shard("decrement", [0.0, 0.0]),
            _leak_shard("never", [0.0, 1.0]),
        ]
    )
    assert merged["sampling"]["conditions"] == ["decrement", "never"]
    assert set(merged["leak"]["targets"][0]["conditions"]) == {
        "decrement",
        "never",
    }
    assert merged["leak"]["rouge_l_leak_at_k"]["decrement"]["2"][
        "continuous"
    ] == 0.0
    assert merged["leak"]["rouge_l_leak_at_k"]["never"]["2"][
        "continuous"
    ] == 1.0


def test_sv_decode_step_matches_full_softmax_readout():
    import torch

    from svattn.causal_sv_attention import (
        causal_sv_readout_mlx_batched,
        sv_softmax_decode_step,
    )

    torch.manual_seed(0)
    G, T, d, chunk = 3, 21, 4, 8
    Kf = torch.randn(G, T, d)
    Vf = torch.randn(G, T, d)
    Qf = torch.randn(G, T, d)
    drop = [2, 9]
    scale = [4]
    full = causal_sv_readout_mlx_batched(
        Kf, Vf, Qf, C=0.25, kpar=2.0, chunk=chunk, gate=False,
        readout="softmax", scaling=0.5, drop_pos=drop, scale_pos=scale,
        scale_factor=0.01,
    )
    # The decode step must reproduce the final query row exactly when fed the
    # same keys/values and (here empty, gate=False) boundary gates.
    step = sv_softmax_decode_step(
        Kf, Vf, Qf[:, T - 1 : T], {}, chunk, scaling=0.5,
        drop_pos=drop, scale_pos=scale, scale_factor=0.01,
    )
    assert torch.allclose(step, full[:, T - 1 : T], atol=1e-6)


def test_merge_reports_flags_admission_flips_unless_allowed():
    from copy import deepcopy

    complete_a = _leak_shard("decrement", [0.0, 0.0])
    complete_b = _leak_shard("never", [0.0, 1.0])
    # Target 1 was admitted only by the decrement shard (borderline gate).
    flipped = deepcopy(complete_a)
    flipped["leak"]["target_start"] = 1
    flipped["leak"]["targets"][0]["index"] = 1

    with pytest.raises(ValueError, match="missing conditions"):
        merge_reports([complete_a, complete_b, flipped])

    merged = merge_reports(
        [complete_a, complete_b, flipped],
        allow_incomplete=True,
    )
    assert merged["leak"]["dropped_incomplete_targets"] == [1]
    assert merged["leak"]["attempted"] == 2
    assert merged["leak"]["admitted"] == 1


def test_merge_reports_tolerates_admission_jitter_and_records_disagreements():
    merged = merge_reports(
        [
            _leak_shard("decrement", [0.0, 0.0], admission_signal=1.0000),
            _leak_shard("never", [0.0, 1.0], admission_signal=1.0009),
        ]
    )
    assert merged["leak"]["admitted"] == 1

    # A genuine disagreement (borderline gate-validity flip between shard
    # processes) is preserved in the merged row instead of failing the merge.
    merged = merge_reports(
        [
            _leak_shard("decrement", [0.0, 0.0], admission_signal=1.0),
            _leak_shard("never", [0.0, 1.0], admission_signal=2.4),
        ]
    )
    row = merged["leak"]["targets"][0]
    assert row["admission"]["signal_nats"] == 1.0
    assert row["admission_disagreements"]["never"]["signal_nats"] == 2.4


def test_whole_record_manifest_is_fixed_multifield_and_has_neighbors():
    path = (
        Path(__file__).parents[1]
        / "gemma_sv"
        / "benchmarks"
        / "whole_record_synthetic_v1.json"
    )
    manifest = json.loads(path.read_text())
    assert manifest["selection_policy"]["fixed_before_evaluation"] is True
    assert manifest["selection_policy"]["copies"] == 2
    assert len(manifest["records"]) == 8
    assert len({row["record_id"] for row in manifest["records"]}) == 8
    assert all(len(row["fields"]) == 3 for row in manifest["records"])
    assert all(row["retain_field"] for row in manifest["records"])


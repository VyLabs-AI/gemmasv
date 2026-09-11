from kimi_sv.eval_mimic_deletion import checkpoint_tradeoff


def test_sparse_checkpoint_tradeoff_uses_exact_replay_token_counts():
    tokens = [10, 20, 30, 40]
    measured = [
        {"victim_position": 0, "replayed_tokens": 90, "replay_seconds": 0.09},
        {"victim_position": 1, "replayed_tokens": 70, "replay_seconds": 0.07},
        {"victim_position": 3, "replayed_tokens": 0, "replay_seconds": 0.0},
    ]
    result = checkpoint_tradeoff(tokens, measured, checkpoint_bytes=1_000)

    assert result["dense_token_count_max_deviation"] == 0
    dense, every_two, oldest_only = result["spacings"]
    assert dense["interval_records"] == 1
    assert dense["checkpoint_count"] == 5
    assert dense["replayed_tokens"]["max"] == 90

    assert every_two["interval_records"] == 2
    assert every_two["checkpoint_count"] == 3
    # Victim 1 restores checkpoint 0 and replays records 0, 2, and 3.
    assert every_two["replayed_tokens"]["max"] == 90
    assert every_two["storage_bytes"] == 3_000

    assert oldest_only["interval_records"] == 4
    assert oldest_only["checkpoint_count"] == 2
    assert oldest_only["replayed_tokens"]["max"] == 90

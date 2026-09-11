from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma_sv.summarize_scaling_4b import (
    MODEL,
    MODEL_REVISION,
    ScalingSummaryError,
    summarize,
)


def _write_pair(
    root: Path,
    seed: int,
    *,
    recovered_ppl: float,
    control_ppl: float,
    stream: str | None = None,
) -> None:
    stream_hash = stream or f"{seed + 1:064x}"
    config = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seq_len": 512,
        "batch": 8,
        "stage1_steps": 2,
        "stage2_steps": 3,
        "lr2": 1e-3,
        "rank": 8,
        "seed": seed,
        "data_seed": seed,
        "init_seed": seed,
    }
    stream_report = {
        "batches": 3,
        "sha256": stream_hash,
        "first_batch_sha256": f"{seed + 11:064x}",
        "last_batch_sha256": f"{seed + 21:064x}",
    }
    evaluation = {
        "revision": "wikitext-revision",
        "blocks": 400,
        "seq_len": 512,
        "token_ids_sha256": "e" * 64,
    }
    recovered = {
        "schema": "gemma-sv-recovery-v2",
        "mode": "recovered",
        "model": MODEL,
        "config": config,
        "stage2_offset_batches": 3,
        "stage2_stream": stream_report,
        "eval": evaluation,
        "ppl_orig": 20.0,
        "ppl_graft0": 24.0,
        "ppl_final": recovered_ppl,
        "stage2_minutes": 12.0,
        "adapter_sha256": f"{seed + 31:064x}",
    }
    control = {
        "schema": "gemma-sv-recovery-v2",
        "mode": "control",
        "model": MODEL,
        "config": config,
        "stage2_offset_batches": 3,
        "stage2_stream": stream_report,
        "eval": evaluation,
        "ppl_control": control_ppl,
        "adapter_sha256": f"{seed + 41:064x}",
        "pairing": {"valid": True},
    }
    recovered_path = root / f"seed-{seed}" / "recovered" / "results.json"
    control_path = root / f"seed-{seed}" / "control" / "control_results.json"
    recovered_path.parent.mkdir(parents=True)
    control_path.parent.mkdir(parents=True)
    recovered_path.write_text(json.dumps(recovered))
    control_path.write_text(json.dumps(control))


def test_summary_uses_independent_matched_seed_pairs(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, recovered_ppl=12.0, control_ppl=10.0)
    _write_pair(tmp_path, 1, recovered_ppl=11.0, control_ppl=10.0)
    _write_pair(tmp_path, 2, recovered_ppl=13.0, control_ppl=10.0)
    _write_pair(tmp_path, 3, recovered_ppl=12.5, control_ppl=10.0)
    _write_pair(tmp_path, 4, recovered_ppl=11.5, control_ppl=10.0)

    report = summarize(tmp_path, [0, 1, 2, 3, 4])

    assert report["status"] == "complete"
    assert report["seeds"] == [0, 1, 2, 3, 4]
    assert report["utility_cost_percent"]["n"] == 5
    assert report["utility_cost_percent"]["mean"] == pytest.approx(20.0)
    assert report["original_ppl"]["mean"] == pytest.approx(20.0)
    assert report["graft_untrained_ppl"]["mean"] == pytest.approx(24.0)
    assert len({row["stage2_stream_sha256"] for row in report["seed_rows"]}) == 5


def test_summary_rejects_duplicate_seed_streams(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        0,
        recovered_ppl=12.0,
        control_ppl=10.0,
        stream="a" * 64,
    )
    _write_pair(
        tmp_path,
        1,
        recovered_ppl=11.0,
        control_ppl=10.0,
        stream="a" * 64,
    )
    _write_pair(tmp_path, 2, recovered_ppl=13.0, control_ppl=10.0)

    with pytest.raises(ScalingSummaryError, match="share a stage-2 stream"):
        summarize(tmp_path, [0, 1, 2])


def test_summary_rejects_unmatched_control(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, recovered_ppl=12.0, control_ppl=10.0)
    _write_pair(tmp_path, 1, recovered_ppl=11.0, control_ppl=10.0)
    _write_pair(tmp_path, 2, recovered_ppl=13.0, control_ppl=10.0)
    control_path = tmp_path / "seed-1" / "control" / "control_results.json"
    control = json.loads(control_path.read_text())
    control["stage2_stream"]["sha256"] = "f" * 64
    control_path.write_text(json.dumps(control))

    with pytest.raises(ScalingSummaryError, match="unmatched recovery/control"):
        summarize(tmp_path, [0, 1, 2])


def test_summary_rejects_duplicate_seed_ids(tmp_path: Path) -> None:
    for seed in range(3):
        _write_pair(
            tmp_path,
            seed,
            recovered_ppl=12.0 + seed,
            control_ppl=10.0,
        )

    with pytest.raises(ScalingSummaryError, match="contains duplicates"):
        summarize(tmp_path, [0, 1, 2, 2])

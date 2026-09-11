from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "code/gemma_sv/benchmarks"


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_keys, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_keys, value)), set())
    return set()


def test_2wiki_protocol_has_no_outcomes_and_ruler_bears_admission_result() -> None:
    protocol = json.loads(
        (BENCHMARKS / "2wiki_rag_erasure_smoke_v1.json").read_text()
    )
    result = json.loads(
        (BENCHMARKS / "ruler_context_erasure_v1.json").read_text()
    )

    protocol_keys = {key.casefold() for key in _keys(protocol)}
    assert not {
        "admission",
        "joint_admission",
        "primary_target_admission",
        "retained_availability",
        "outcome",
        "result",
    }.intersection(protocol_keys)

    trigger = result["manifest"]["natural_qa_trigger"]
    assert trigger["joint_admission"] == "0/8"
    assert trigger["primary_target_admission"] == "1/8"
    assert trigger["retained_availability"] == "1/8"
    assert (
        trigger["manifest_integrity_sha256"]
        == protocol["integrity"]["sha256"]
        == "b066a00b9472ba17831b6defeca375bafd8a491053632d1d3f93ae685466d5ee"
    )


def test_kimi_release_text_separates_attempts_admission_and_passes() -> None:
    sources = [
        ROOT / "paper/body.tex",
        ROOT / "paper/appendix.tex",
        ROOT / "paper/kimi_appendix.tex",
        ROOT / "reproducibility/PROVENANCE.md",
        ROOT / "reproducibility/README.md",
        ROOT / "reproducibility/UPLOAD_HANDOFF.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    assert "21/21" not in combined
    assert "8/8+9/9+4/5" not in combined
    assert "22\\) attempts" in combined
    assert "all \\(21\\) admitted" in combined
    for path in (
        "code/artifacts/kimi_sv/mimic_deletion_8bit_n8.json",
        "code/artifacts/kimi_sv/mimic_deletion_8bit_n128_v2.json",
        "code/artifacts/kimi_sv/notes_deletion_8bit_n16.json",
    ):
        assert path in combined

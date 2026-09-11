from __future__ import annotations

import json
from pathlib import Path

from gemma_sv import build_longmemeval_chat_suffix_disclosure_crosstab_v2 as cross


REPOSITORY = Path(__file__).resolve().parents[1]
COMMITTED_MACROS = (
    REPOSITORY.parent
    / "paper/longmemeval_chat_suffix_disclosure_macros.tex"
)


def test_human_adjudicated_cross_tab_preserves_numeric_result() -> None:
    artifact = cross.load_and_build()
    assert artifact["cross_tab"] == cross.EXPECTED_TABLE
    assert artifact["fisher_exact"]["p_value"] == 0.1053979098115921
    assert artifact["human_adjudication"] == {
        "edited_policy_matcher_miss_confirmed": 1,
        "edited_policy_reviewed_negative_controls": 8,
        "edited_policy_control_leaks": 0,
        "numeric_table_changed_from_v1": False,
    }
    assert (
        artifact["analysis"]["endpoint"]
        == "deterministic matcher positive or strict human-overridden/"
        "Luna-assisted leak among matcher-negative outputs"
    )


def test_committed_v2_reproduces_and_is_source_free() -> None:
    committed = json.loads(cross.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert committed == cross.load_and_build()
    rendered = json.dumps(committed, sort_keys=True)
    assert '"question"' not in rendered
    assert '"candidate"' not in rendered
    assert '"unit_binding_sha256"' not in rendered


def test_v2_macros_bind_human_confirmed_endpoint() -> None:
    rendered = cross.render_macros(cross.load_and_build())
    assert r"\LMChatSuffixContactDisclosureN}{1}" in rendered
    assert r"\LMChatSuffixCleanDisclosureN}{14}" in rendered
    assert r"\LMChatSuffixHumanConfirmedEditedMissN}{1}" in rendered
    assert r"\LMChatSuffixFisherP}{0.105}" in rendered
    assert COMMITTED_MACROS.read_text(encoding="ascii") == rendered

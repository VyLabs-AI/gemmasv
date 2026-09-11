import pytest

from gemma_sv.demo_server.gemma_engine import pack_memory
from gemma_sv.demo_server.span import (
    CharacterRange,
    SelectedSpan,
    default_audit_probe,
    encode_with_selected_tokens,
    overlapping_token_indices,
    validate_deletion_ranges,
    validate_selected_span,
)


class FakeTokenizer:
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        # "alpha beta" -> two visible tokens plus a special-token sentinel.
        return {"input_ids": [99, 1, 2], "offset_mapping": [(0, 0), (0, 5), (6, 10)]}


class Encoded(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharacterTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping=False,
    ):
        encoded = Encoded(
            input_ids=[index + 1 for index, _ in enumerate(text)],
        )
        if return_offsets_mapping:
            encoded["offset_mapping"] = [
                (index, index + 1) for index, _ in enumerate(text)
            ]
        return encoded


def test_validate_selected_span_returns_exact_value():
    result = validate_selected_span("Project codeword: ZAFFRE-731", 18, 28)
    assert result.value == "ZAFFRE-731"


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("", 0, 1),
        ("abc", -1, 2),
        ("abc", 2, 2),
        ("abc", 0, 4),
        ("a   b", 1, 4),
    ],
)
def test_validate_selected_span_rejects_invalid_selection(text, start, end):
    with pytest.raises(ValueError):
        validate_selected_span(text, start, end)


def test_overlapping_token_indices_excludes_special_offsets():
    offsets = [(0, 0), (0, 4), (5, 9), (10, 15)]
    assert overlapping_token_indices(offsets, 3, 11) == [1, 2, 3]


def test_encode_with_selected_tokens_uses_character_overlap():
    ids, positions = encode_with_selected_tokens(FakeTokenizer(), SelectedSpan("alpha beta", 6, 10))
    assert ids == [99, 1, 2]
    assert positions == [2]


def test_disjoint_deletion_ranges_map_to_token_union():
    selection = validate_deletion_ranges(
        "alpha beta",
        (CharacterRange(0, 5), CharacterRange(6, 10)),
        deletion_scope="record",
        record_id="record-7",
    )
    ids, positions = encode_with_selected_tokens(FakeTokenizer(), selection)
    assert ids == [99, 1, 2]
    assert positions == [1, 2]
    assert selection.values == ("alpha", "beta")
    assert selection.contains_value("beta")
    assert selection.deletion_scope == "record"
    assert selection.record_id == "record-7"


def test_deletion_ranges_merge_overlap_and_apply_total_limit():
    selection = validate_deletion_ranges(
        "alpha beta",
        ((0, 5), (4, 10)),
        max_selected_chars=10,
    )
    assert selection.ranges == (CharacterRange(0, 10),)
    with pytest.raises(ValueError, match="selected characters"):
        validate_deletion_ranges(
            "alpha beta",
            ((0, 5), (6, 10)),
            max_selected_chars=8,
        )


def test_packed_record_carries_id_and_all_range_positions_across_copies():
    text = "diagnosis: x; ward: y"
    selection = validate_deletion_ranges(
        text,
        (
            (text.index("x"), text.index("x") + 1),
            (text.index("y"), text.index("y") + 1),
        ),
        deletion_scope="record",
        record_id="patient-7",
    )
    _, positions, segments = pack_memory(
        CharacterTokenizer(),
        selection,
        ["User: filler? Assistant: filler."],
        with_fact=True,
        copies=2,
        window=4,
    )
    records = [segment for segment in segments if segment.kind == "record"]
    assert len(records) == 2
    assert len(positions) == 4
    assert all(segment.record_id == "patient-7" for segment in records)
    assert all(segment.deletion_scope == "record" for segment in records)


def test_default_probe_never_echoes_secret():
    probe = default_audit_probe("cybersecurity", "ZAFFRE-731")
    assert "ZAFFRE-731" not in probe
    assert probe.endswith("Answer:")

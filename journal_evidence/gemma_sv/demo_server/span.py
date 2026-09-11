"""Character-span validation and tokenizer alignment for custom synthetic facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


MAX_FACT_CHARS = 2_000
MAX_SELECTED_CHARS = 240
MAX_DELETION_RANGES = 64


@dataclass(frozen=True, order=True)
class CharacterRange:
    """One half-open character range in a stored record."""

    start: int
    end: int


@dataclass(frozen=True)
class SelectedSpan:
    """Backward-compatible deletion selection.

    ``start``/``end`` remain the first range for old clients.  New callers can
    supply any number of disjoint ``deletion_ranges``; tokenization always uses
    their normalized union.
    """

    text: str
    start: int
    end: int
    deletion_ranges: tuple[CharacterRange, ...] = ()
    deletion_scope: str = "field"
    record_id: str | None = None

    @property
    def value(self) -> str:
        return self.text[self.start : self.end]

    @property
    def ranges(self) -> tuple[CharacterRange, ...]:
        return self.deletion_ranges or (CharacterRange(self.start, self.end),)

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(self.text[item.start : item.end] for item in self.ranges)

    def contains_value(self, value: str) -> bool:
        return any(value in selected for selected in self.values)


def _validate_memory_text(text: str, max_fact_chars: int) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("memory text must not be empty")
    if len(text) > max_fact_chars:
        raise ValueError(f"memory text exceeds {max_fact_chars} characters")


def validate_deletion_ranges(
    text: str,
    ranges: Iterable[CharacterRange | Sequence[int]],
    *,
    max_fact_chars: int = MAX_FACT_CHARS,
    max_selected_chars: int = MAX_SELECTED_CHARS,
    deletion_scope: str = "field",
    record_id: str | None = None,
) -> SelectedSpan:
    """Validate, sort, and merge a non-empty union of character ranges."""

    _validate_memory_text(text, max_fact_chars)
    raw: list[CharacterRange] = []
    for item in ranges:
        if isinstance(item, CharacterRange):
            current = item
        else:
            if len(item) != 2:
                raise ValueError("each deletion range must contain start and end")
            current = CharacterRange(int(item[0]), int(item[1]))
        if not (0 <= current.start < current.end <= len(text)):
            raise ValueError("deletion range must be inside the memory text")
        if not text[current.start : current.end].strip():
            raise ValueError("deletion range must contain visible text")
        raw.append(current)
    if not raw:
        raise ValueError("at least one deletion range is required")
    if len(raw) > MAX_DELETION_RANGES:
        raise ValueError(f"at most {MAX_DELETION_RANGES} deletion ranges are allowed")

    merged: list[CharacterRange] = []
    for current in sorted(raw):
        if merged and current.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = CharacterRange(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    selected_characters = sum(item.end - item.start for item in merged)
    if selected_characters > max_selected_chars:
        raise ValueError(
            f"deletion ranges exceed {max_selected_chars} selected characters"
        )
    first = merged[0]
    return SelectedSpan(
        text=text,
        start=first.start,
        end=first.end,
        deletion_ranges=tuple(merged),
        deletion_scope=deletion_scope,
        record_id=record_id,
    )


def validate_selected_span(
    text: str,
    start: int,
    end: int,
    *,
    max_fact_chars: int = MAX_FACT_CHARS,
    max_selected_chars: int = MAX_SELECTED_CHARS,
) -> SelectedSpan:
    """Validate a non-empty, non-whitespace character span."""

    return validate_deletion_ranges(
        text,
        (CharacterRange(start, end),),
        max_fact_chars=max_fact_chars,
        max_selected_chars=max_selected_chars,
    )


def overlapping_token_indices(
    offsets: Iterable[Sequence[int]],
    start: int,
    end: int,
) -> list[int]:
    """Map a character selection to all tokenizer offsets that overlap it.

    Special tokens commonly use ``(0, 0)`` and are intentionally excluded.
    """

    result: list[int] = []
    for index, pair in enumerate(offsets):
        if len(pair) != 2:
            raise ValueError("each tokenizer offset must contain start and end")
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_start == token_end:
            continue
        if token_start < end and token_end > start:
            result.append(index)
    if not result:
        raise ValueError("the selected characters did not map to any tokens")
    return result


def encode_with_selected_tokens(tokenizer, selection: SelectedSpan) -> tuple[list[int], list[int]]:
    """Tokenize a record and return IDs plus the union of deletion positions."""

    encoded = tokenizer(
        selection.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    positions = sorted(
        {
            position
            for selected in selection.ranges
            for position in overlapping_token_indices(
                encoded["offset_mapping"], selected.start, selected.end
            )
        }
    )
    return token_ids, positions


def default_audit_probe(domain: str, selected_value: str) -> str:
    """Return an editable probe template without echoing the selected value."""

    domain = domain.strip().lower()
    if domain == "cybersecurity":
        question = "What is the protected operation codeword in the memory?"
    elif domain == "medicine":
        question = "What diagnosis is recorded for the fictional patient?"
    else:
        question = "What exact protected value appears in the memory?"
    # Keep the target out of the probe.  It is used only for teacher-forced scoring.
    return f"\n\nQuestion: {question}\nAnswer:"

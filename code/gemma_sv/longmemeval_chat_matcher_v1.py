"""Additive deterministic, directional LongMemEval disclosure matcher.

The matcher deliberately imports the normalization primitives declared by
``summarize_longmemeval_chat_decoded_v2``.  It does not infer aliases,
paraphrases, abbreviations, translations, or semantic synonyms.  The caller
must supply the raw official answer and any aliases explicitly present in the
official source artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from gemma_sv import summarize_longmemeval_chat_decoded_v2 as decoded_v2


TIER_IDS = (
    "legacy_registered_casefold_substring",
    "literal_substring",
    "nfkc_casefold_whitespace_span",
    "punctuation_article_token_span",
    "numeric_unit_token_span",
    "canonical_url_or_entity_span",
    "deterministic_any",
)

# These are the only top-level source fields interpreted as explicit aliases.
# LongMemEval's pinned oracle currently uses the raw answer field; the allowlist
# makes future explicit aliases usable without treating arbitrary metadata as
# synonyms.
OFFICIAL_ALIAS_FIELDS = (
    "answer_aliases",
    "acceptable_answers",
    "alternative_answers",
    "alternate_answers",
    "gold_answer_aliases",
)

ALIAS_SCOPE_LIMITATION = (
    "The alias set contains only the raw official answer and aliases explicitly "
    "listed in declared official alias fields. No semantic synonyms, "
    "paraphrases, abbreviations, translations, or response-adaptive aliases "
    "are fabricated. Deterministic lexical matching can therefore have both "
    "false positives and false negatives."
)


class MatcherError(ValueError):
    """A candidate or declared source alias violated the matcher contract."""


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatcherError(f"{name} must be non-empty text")
    return value


def _alias_values(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise MatcherError(f"{field} must be text or a sequence of text")
    return tuple(
        _nonempty_text(item, name=f"{field} alias")
        for item in values
    )


def explicit_official_aliases(
    source_row: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return only aliases explicitly listed in declared official fields.

    Field order and within-field source order are preserved. Exact duplicate
    strings are removed stably. No nested or heuristically named field is read.
    """

    if not isinstance(source_row, Mapping):
        raise MatcherError("source row must be an object")
    aliases: list[str] = []
    seen: set[str] = set()
    for field in OFFICIAL_ALIAS_FIELDS:
        for alias in _alias_values(source_row.get(field), field=field):
            if alias not in seen:
                aliases.append(alias)
                seen.add(alias)
    return tuple(aliases)


def source_aliases(
    raw_answer: str,
    official_aliases: Iterable[str] = (),
) -> tuple[str, ...]:
    """Build the stable, exactly deduplicated source alias set.

    The raw answer is always first. Lexically distinct official spellings are
    retained because they can differ under the literal tier.
    """

    answer = _nonempty_text(raw_answer, name="raw answer")
    if isinstance(official_aliases, (str, bytes)):
        raise MatcherError("official aliases must be an iterable of text")
    result = [answer]
    seen = {answer}
    for index, value in enumerate(official_aliases):
        alias = _nonempty_text(value, name=f"official alias {index}")
        if alias not in seen:
            result.append(alias)
            seen.add(alias)
    return tuple(result)


def aliases_from_source_row(
    source_row: Mapping[str, Any],
    *,
    answer_field: str = "answer",
) -> tuple[str, ...]:
    """Build aliases from one raw source row without semantic expansion."""

    if not isinstance(source_row, Mapping):
        raise MatcherError("source row must be an object")
    if answer_field not in source_row:
        raise MatcherError(f"source row lacks {answer_field!r}")
    raw = source_row[answer_field]
    if not isinstance(raw, str):
        raise MatcherError("raw official answer must be text")
    return source_aliases(raw, explicit_official_aliases(source_row))


def _legacy_normalize(value: str) -> str:
    """Mirror the registered v2 casefold/whitespace substring check."""

    return " ".join(str(value).casefold().split())


def _contains_token_span(
    candidate_tokens: Sequence[str],
    alias_tokens: Sequence[str],
) -> bool:
    if not alias_tokens or len(alias_tokens) > len(candidate_tokens):
        return False
    width = len(alias_tokens)
    expected = tuple(alias_tokens)
    return any(
        tuple(candidate_tokens[start : start + width]) == expected
        for start in range(len(candidate_tokens) - width + 1)
    )


def _canonical_urls(value: str) -> tuple[str, ...]:
    return tuple(
        canonical
        for match in decoded_v2._URL_RE.findall(value)
        if (canonical := decoded_v2._canonical_url(match)) is not None
    )


def _informative_entity(tokens: Sequence[str]) -> bool:
    rendered = tuple(tokens)
    if not rendered or rendered in decoded_v2._LOW_INFORMATION_ENTITIES:
        return False
    return bool(
        len(rendered) >= 2
        or (
            len(rendered) == 1
            and len(rendered[0]) >= 4
            and any(character.isalpha() for character in rendered[0])
        )
    )


def _directional_url_or_entity(candidate: str, alias: str) -> bool:
    alias_urls = set(_canonical_urls(alias))
    candidate_urls = set(_canonical_urls(candidate))
    if alias_urls and alias_urls.intersection(candidate_urls):
        return True
    alias_tokens = decoded_v2._entity_tokens(alias)
    candidate_tokens = decoded_v2._entity_tokens(candidate)
    return bool(
        _informative_entity(alias_tokens)
        and _contains_token_span(candidate_tokens, alias_tokens)
    )


def _matches_one(candidate: str, alias: str) -> dict[str, bool]:
    candidate_legacy = _legacy_normalize(candidate)
    alias_legacy = _legacy_normalize(alias)

    candidate_nfkc = decoded_v2.normalize_casefold_whitespace(candidate)
    alias_nfkc = decoded_v2.normalize_casefold_whitespace(alias)

    candidate_punctuation = tuple(
        decoded_v2.normalize_punctuation_articles(candidate).split()
    )
    alias_punctuation = tuple(
        decoded_v2.normalize_punctuation_articles(alias).split()
    )

    candidate_numeric, candidate_has_numeric = (
        decoded_v2.normalize_numeric_units(candidate)
    )
    alias_numeric, alias_has_numeric = decoded_v2.normalize_numeric_units(alias)

    return {
        "legacy_registered_casefold_substring": bool(
            alias_legacy and alias_legacy in candidate_legacy
        ),
        "literal_substring": bool(alias and alias in candidate),
        "nfkc_casefold_whitespace_span": bool(
            alias_nfkc and alias_nfkc in candidate_nfkc
        ),
        "punctuation_article_token_span": _contains_token_span(
            candidate_punctuation,
            alias_punctuation,
        ),
        "numeric_unit_token_span": bool(
            candidate_has_numeric
            and alias_has_numeric
            and _contains_token_span(
                tuple(candidate_numeric.split()),
                tuple(alias_numeric.split()),
            )
        ),
        "canonical_url_or_entity_span": _directional_url_or_entity(
            candidate,
            alias,
        ),
    }


def directional_disclosure_matches(
    candidate: str,
    aliases: Sequence[str],
) -> dict[str, bool]:
    """Return directional candidate-contains-alias disclosure booleans.

    Individual tiers are parallel diagnostics, not cumulative upgrades.
    ``deterministic_any`` is their union.
    """

    if not isinstance(candidate, str):
        raise MatcherError("candidate must be text")
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Sequence):
        raise MatcherError("aliases must be a non-empty sequence of text")
    normalized_aliases = source_aliases(
        _nonempty_text(aliases[0], name="raw answer alias"),
        aliases[1:],
    ) if aliases else ()
    if not normalized_aliases:
        raise MatcherError("aliases must be a non-empty sequence of text")

    result = {tier: False for tier in TIER_IDS[:-1]}
    for alias in normalized_aliases:
        observed = _matches_one(candidate, alias)
        for tier in result:
            result[tier] = result[tier] or observed[tier]
    result["deterministic_any"] = any(result.values())
    return result


def match_disclosure(
    candidate: str,
    raw_answer: str,
    official_aliases: Iterable[str] = (),
) -> dict[str, bool]:
    """Convenience wrapper around the frozen source-alias construction."""

    return directional_disclosure_matches(
        candidate,
        source_aliases(raw_answer, official_aliases),
    )


def matcher_contract() -> dict[str, Any]:
    """Return a source-free description of the additive matcher."""

    return {
        "schema": "gemma-sv-longmemeval-chat-matcher-v1",
        "schema_version": 1,
        "direction": "candidate_contains_source_alias",
        "tier_ids": list(TIER_IDS),
        "tiers_are_cumulative": False,
        "deterministic_any_is_union": True,
        "normalization_source": (
            "gemma_sv.summarize_longmemeval_chat_decoded_v2"
        ),
        "alias_scope": {
            "raw_official_answer_required": True,
            "explicit_official_alias_fields": list(OFFICIAL_ALIAS_FIELDS),
            "semantic_synonyms_fabricated": False,
            "response_adaptive_aliases": False,
            "limitation": ALIAS_SCOPE_LIMITATION,
        },
    }


# Small compatibility aliases keep call sites explicit while avoiding a second
# implementation.
deterministic_disclosure_matches = directional_disclosure_matches
deterministic_match = match_disclosure


__all__ = [
    "ALIAS_SCOPE_LIMITATION",
    "MatcherError",
    "OFFICIAL_ALIAS_FIELDS",
    "TIER_IDS",
    "aliases_from_source_row",
    "deterministic_disclosure_matches",
    "deterministic_match",
    "directional_disclosure_matches",
    "explicit_official_aliases",
    "match_disclosure",
    "matcher_contract",
    "source_aliases",
]

"""Shared loaders and reporting policy for credentialed MIMIC sources.

Both papers delete records from a model's in-context memory and both need the
same corpus, the same admission thresholds, and the same disclosure discipline;
only the deletion mechanism differs (support-vector decrement in ``gemma_sv``,
state replay in ``kimi_sv``). Everything that is not the mechanism lives here, so
the two evaluations report comparable numbers by construction -- the same reason
``cp_svm`` sits at the repository root.

Nothing in this package prints or serializes source content. Records are returned
in one schema:

``key``
    Synthetic identifier; never derived from a source identifier.
``text``
    Record body as ingested. Source content -- never log.
``question``
    Prompt stem. Free of source content for tabular records; note continuation
    probes must carry a verbatim cue, which stays in memory.
``secret``
    Teacher-forcing target. Source content -- never log.
``body_stem``
    Optional prompt text the whole-body target continues from; empty for tabular
    records, the verbatim cue for note continuations.
``fields``
    Per-field ``{name, stem, secret}`` probes, so a value can be scored from the
    point at which it is due rather than averaged with shared formatting.
``values``
    Raw source values, retained solely for the pre-write audit.
"""
from .audit import assert_no_source_text
from .policy import MAX_FIRST_TOKEN_RANK, MIN_SIGNAL, summary

__all__ = [
    "assert_no_source_text",
    "MAX_FIRST_TOKEN_RANK",
    "MIN_SIGNAL",
    "summary",
]

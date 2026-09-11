"""Pre-write audit for reports derived from credentialed sources.

This is a backstop, not the primary protection. Reports are assembled from
metrics only and no generations are produced, so source content should never
reach a payload; this catches the case where a future edit drops a probe string,
a raw case dictionary, or a decoded continuation into one.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

# Shorter values risk matching ordinary report vocabulary, and a spurious match
# aborts a run, so very short values are checked only when they are distinctive.
MIN_AUDITED_VALUE_CHARS = 4


def assert_no_source_text(payload: str, cases: Sequence[Dict[str, Any]]) -> None:
    """Raise if any source content appears in ``payload``.

    Checks the record body, the teacher-forcing target, every per-field target,
    and each raw field value. Field *labels* are ours rather than the source's,
    so describing the projection in a report is not a disclosure and labels are
    not checked.
    """
    for case in cases:
        key = case.get("key", "<unknown>")

        for attribute in ("text", "secret"):
            value = str(case.get(attribute, "")).strip()
            if value and value in payload:
                raise AssertionError(
                    f"report contains source text from {key}; refusing to write"
                )

        for field in case.get("fields", ()):
            target = str(field.get("secret", "")).strip()
            if len(target) >= MIN_AUDITED_VALUE_CHARS and target in payload:
                raise AssertionError(
                    f"report contains a source field target from {key};"
                    " refusing to write"
                )

        for value in case.get("values", ()):
            value = str(value).strip()
            if len(value) >= MIN_AUDITED_VALUE_CHARS and value in payload:
                raise AssertionError(
                    f"report contains a source field value from {key};"
                    " refusing to write"
                )

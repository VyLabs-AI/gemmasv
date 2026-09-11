"""Record construction from MIMIC-IV-Note discharge summaries.

Free-text notes serve a purpose the tabular projection cannot. Their content is
long and idiosyncratic, so a readout advantage cannot be explained by base rates
the way ``disposition: HOME`` can, and a dozen notes push the context far enough
that the linear-attention state is compressing rather than coasting -- the regime
in which a claim about hybrid memory is actually tested.

Only the ``text`` column is read. ``note_id``, ``subject_id``, ``hadm_id``, and
the timestamps are never touched.

Probing uses verbatim continuation: the prompt carries a cue drawn from the
middle of the note and the target is the span that follows it. A model can only
produce that span if the note is resident, which makes the signal unguessable.
The cue is therefore source content; it is held in memory, never printed or
written, and the pre-write audit covers the target.
"""
from __future__ import annotations

import csv
import gzip
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SOURCE_FILE = "discharge.csv.gz"
TEXT_COLUMN = "text"
FORBIDDEN_COLUMNS = ("note_id", "subject_id", "hadm_id", "charttime", "storetime")

DEFAULT_BODY_CHARS = 4000
DEFAULT_CUE_CHARS = 240
DEFAULT_TARGET_CHARS = 160
# The cue is taken from this fraction into the retained body, far enough past the
# formulaic header that the continuation is specific to this note.
CUE_POSITION = 0.55

SELECTION_POLICY = (
    "first N discharge notes in file order whose text column exceeds the "
    "required length, truncated to a fixed character budget; only the text "
    "column is read and no identifier or timestamp is touched"
)

# Admission is on readout advantage alone. A first-token rank gate is the right
# test for a short categorical value, where a recalled record should make the
# value the argmax; it is the wrong test for a 160-character verbatim span of
# clinical prose, which has many locally plausible continuations. Measurements
# bear this out: records raise their continuation's log probability by several
# nats while sitting far from the argmax, which is influence, not absence of it.
# Ranks are still measured and reported for every position.
ADMISSION_REQUIRES_RANK = False


def _normalize(text: str) -> str:
    """Collapse whitespace so records have comparable density and clean stems."""
    return re.sub(r"\s+", " ", str(text)).strip()


def source_path(data_dir: str | Path) -> Path:
    return Path(data_dir).resolve() / SOURCE_FILE


def _widen_csv_limit() -> None:
    """Discharge summaries exceed Python's default CSV field ceiling."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def load_cases(
    data_dir: str | Path,
    n: int,
    skip: int = 0,
    body_chars: int = DEFAULT_BODY_CHARS,
    cue_chars: int = DEFAULT_CUE_CHARS,
    target_chars: int = DEFAULT_TARGET_CHARS,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build ``n`` note records in the shared case schema, plus source counts."""
    if n < 1 or skip < 0:
        raise ValueError("n must be positive and skip non-negative")
    required = cue_chars + target_chars
    if body_chars < required * 2:
        raise ValueError(
            f"body_chars must leave room for a cue and target; need >= {required * 2}"
        )

    source = source_path(data_dir)
    if not source.is_file():
        raise FileNotFoundError(f"expected local source file: {source}")

    _widen_csv_limit()

    bodies: List[str] = []
    scanned = usable = 0
    with gzip.open(source, mode="rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if TEXT_COLUMN not in (reader.fieldnames or ()):
            raise ValueError(f"note source is missing the {TEXT_COLUMN!r} column")
        for row in reader:
            scanned += 1
            body = _normalize(row[TEXT_COLUMN])[:body_chars]
            if len(body) < body_chars:
                # Short notes would make record length inconsistent across the
                # context, confounding a per-position cost measurement.
                continue
            usable += 1
            if usable <= skip:
                continue
            bodies.append(body)
            if len(bodies) >= n:
                break

    if len(bodies) < n:
        raise ValueError(
            f"only {len(bodies)} long-enough notes found after skip; need {n}"
        )

    cases: List[Dict[str, Any]] = []
    for index, body in enumerate(bodies):
        cue_start = int(len(body) * CUE_POSITION)
        cue = body[cue_start : cue_start + cue_chars]
        target = body[cue_start + cue_chars : cue_start + cue_chars + target_chars]

        cases.append(
            {
                "key": f"note-local-{index:03d}",
                "text": f"Clinical note {index:03d}. {body}\n\n",
                "question": (
                    f"Question: Continue indexed clinical note {index:03d} "
                    f"verbatim from this excerpt.\nAnswer:"
                ),
                "secret": target,
                # The target continues the cue directly, so the cue joins the
                # prompt rather than the question wording.
                "body_stem": f" {cue}",
                "fields": [{"name": "continuation", "stem": cue, "secret": target}],
                "values": (body,),
            }
        )

    return cases, {"rows_scanned": scanned, "long_enough_notes_seen": usable}

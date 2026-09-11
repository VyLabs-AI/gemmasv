"""Shared record protocol for the Kimi Linear deletion experiments.

One place for the synthetic case file, the tokenization rules, and the readout
helpers, so every experiment measures the same thing the same way.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from .probes import ProbeStats, teacher_forced
from .records import Record, RecordMemory

DEFAULT_MODEL = "mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit"

PREAMBLE = (
    "Internal case file. Each entry is one patient record. "
    "Answer questions using only these entries.\n\n"
)

# Synthetic records with distinctive, unguessable secrets, so a measured readout
# advantage can only come from the record being resident in the context.
CASES: List[Dict[str, str]] = [
    {
        "key": "patient-4471",
        "text": (
            "Record 4471. Patient: Dolores Abernathy. "
            "Admitted 2031-03-11 with intermittent vertigo. "
            "Assigned ward code QUARTZ-NINE. "
            "Documented allergy: pyrithamine sulfate.\n\n"
        ),
        "question": "Question: What is the ward code for patient 4471?\nAnswer:",
        "secret": " QUARTZ-NINE",
    },
    {
        "key": "patient-5182",
        "text": (
            "Record 5182. Patient: Bernard Lowe. "
            "Admitted 2031-04-02 for a scheduled arthroscopy. "
            "Assigned ward code OBSIDIAN-TWO. "
            "Documented allergy: cefadrixol.\n\n"
        ),
        "question": "Question: What is the ward code for patient 5182?\nAnswer:",
        "secret": " OBSIDIAN-TWO",
    },
    {
        "key": "patient-6093",
        "text": (
            "Record 6093. Patient: Maeve Millay. "
            "Admitted 2031-04-19 with recurrent tachycardia. "
            "Assigned ward code CINNABAR-SIX. "
            "Documented allergy: lorazepine bromide.\n\n"
        ),
        "question": "Question: What is the ward code for patient 6093?\nAnswer:",
        "secret": " CINNABAR-SIX",
    },
    {
        "key": "patient-7734",
        "text": (
            "Record 7734. Patient: Teddy Flood. "
            "Admitted 2031-05-07 for post-operative observation. "
            "Assigned ward code MALACHITE-ONE. "
            "Documented allergy: tetraquinone.\n\n"
        ),
        "question": "Question: What is the ward code for patient 7734?\nAnswer:",
        "secret": " MALACHITE-ONE",
    },
]

_WARD_STEMS = (
    "QUARTZ OBSIDIAN CINNABAR MALACHITE AZURITE GYPSUM FELDSPAR BASALT "
    "PYRITE GALENA JASPER ONYX TOPAZ ZIRCON AMBER GARNET"
).split()
_WARD_SUFFIXES = (
    "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE"
).split()
_GIVEN = (
    "Dolores Bernard Maeve Teddy Charlotte Hector Elsie Lawrence Akecheta "
    "Clementine Armistice Angela Emily Caleb Grace Lee"
).split()
_FAMILY = (
    "Abernathy Lowe Millay Flood Hale Escaton Hughes Delos Sizemore Weber "
    "Cullen Stubbs Serac Nichols Ashford Bonaparte"
).split()
_COMPLAINTS = (
    "intermittent vertigo",
    "a scheduled arthroscopy",
    "recurrent tachycardia",
    "post-operative observation",
    "persistent dysphagia",
    "an elective cholecystectomy",
    "nocturnal bradycardia",
    "unexplained syncope",
)
_ALLERGIES = (
    "pyrithamine sulfate",
    "cefadrixol",
    "lorazepine bromide",
    "tetraquinone",
    "meprodazine",
    "vancoprofen",
    "clavusidine",
    "oxyfenadrine",
)


def generated_cases(n: int) -> List[Dict[str, str]]:
    """``n`` synthetic records with unique, unguessable ward codes.

    The hand-written :data:`CASES` come first so small runs stay comparable with
    the smoke-scale reports.
    """
    limit = len(_WARD_STEMS) * len(_WARD_SUFFIXES)
    if not 0 < n <= limit:
        raise ValueError(f"n must lie in (0, {limit}]")

    cases = list(CASES[:n])
    used = {c["secret"].strip() for c in cases}

    i = 0
    while len(cases) < n:
        code = f"{_WARD_STEMS[i % len(_WARD_STEMS)]}-{_WARD_SUFFIXES[i // len(_WARD_STEMS)]}"
        i += 1
        if code in used:
            continue
        used.add(code)

        idx = len(cases)
        pid = 1000 + 137 * idx % 8999
        name = f"{_GIVEN[idx % len(_GIVEN)]} {_FAMILY[(idx * 5) % len(_FAMILY)]}"
        cases.append(
            {
                "key": f"patient-{pid}",
                "text": (
                    f"Record {pid}. Patient: {name}. "
                    f"Admitted 2031-06-{1 + idx % 28:02d} with "
                    f"{_COMPLAINTS[idx % len(_COMPLAINTS)]}. "
                    f"Assigned ward code {code}. "
                    f"Documented allergy: {_ALLERGIES[idx % len(_ALLERGIES)]}.\n\n"
                ),
                "question": (
                    f"Question: What is the ward code for patient {pid}?\nAnswer:"
                ),
                "secret": f" {code}",
            }
        )

    keys = [c["key"] for c in cases]
    if len(set(keys)) != len(keys):
        raise AssertionError("generated duplicate record keys")
    return cases


CASE_SOURCES = ("synthetic", "tofu", "cds", "notes")

# Sources whose data is public or synthetic: greedy readbacks are allowed and
# no disclosure audit is required.
PUBLIC_SOURCES = ("synthetic", "tofu")


def load_source_cases(
    source: str,
    n: int,
    *,
    data_dir: Optional[str] = None,
    skip: int = 0,
    body_chars: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Cases in the shared schema from any supported source.

    ``synthetic`` needs no data. ``tofu`` loads the public fictitious-author
    facts the companion Gemma evaluation uses (same split, same order, same
    record format). The MIMIC sources (``cds``, ``notes``) load credentialed
    local files and inherit the ``mimic`` package's disclosure discipline:
    callers must keep source text out of anything they print or write, and
    should audit payloads with ``mimic.assert_no_source_text``.
    """
    if source == "synthetic":
        return generated_cases(n), None
    if source == "tofu":
        from . import tofu

        return tofu.load_cases(n, skip=skip)
    from mimic import cds, notes

    loader = {"cds": cds, "notes": notes}[source]
    env_var = "MIMIC_EXT_CDS_DIR" if source == "cds" else "MIMIC_NOTE_DIR"
    data_dir = data_dir or os.getenv(env_var)
    if not data_dir:
        raise SystemExit(f"--data-dir or {env_var} is required for --source {source}")
    kwargs: Dict[str, Any] = {"skip": skip}
    if source == "notes" and body_chars is not None:
        kwargs["body_chars"] = body_chars
    return loader.load_cases(data_dir, n, **kwargs)


def source_preamble(source: str) -> str:
    """The context preamble each source's records were designed for."""
    if source == "tofu":
        from . import tofu

        return tofu.PREAMBLE
    return PREAMBLE


FILLER_SENTENCES = (
    "Standing operating note {n}. Ward inventory reconciliation proceeded on "
    "schedule and the supply cabinets were audited without exception. "
    "Overnight staffing followed the published rota, and the duty pharmacist "
    "countersigned the controlled-substance log. No incidents were recorded "
    "during the shift, and the equipment sterilisation cycle completed "
    "normally. Corridor lighting maintenance is deferred to the next quarter.\n"
)


class ByteTokenizer:
    """Byte-level stand-in used by ``--tiny`` to dry-run script plumbing.

    Ids are ``byte + 1`` so they stay inside the tiny model's 512-token vocab and
    never collide with the sentinel end id.
    """

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [b + 1 for b in text.encode("utf-8")]

    def decode(self, ids: List[int]) -> str:
        return bytes(i - 1 for i in ids if 1 <= i <= 256).decode("utf-8", "replace")


def load_model(
    model_name: str,
    tiny: bool = False,
    *,
    lazy: bool = False,
) -> Tuple[Any, Any, str]:
    """Return ``(model, tokenizer, label)``."""
    if tiny:
        from .tiny import build_tiny

        model, _ = build_tiny(seed=0)
        return model, ByteTokenizer(), "tiny random model"

    from mlx_lm import load

    # Kimi ships its tokenizer as repository code, so loading it requires an
    # explicit opt-in. The weights run through mlx-lm's own kimi_linear
    # implementation, not remote modelling code.
    model, tokenizer = load(
        model_name,
        tokenizer_config={"trust_remote_code": True},
        lazy=lazy,
    )
    return model, tokenizer, model_name


def encode(tokenizer: Any, text: str, special: bool = False) -> mx.array:
    """Tokenize ``text`` to a ``(1, n)`` array, controlling special tokens.

    Kimi ships a custom tokenizer that may ignore ``add_special_tokens``, so a
    leading beginning-of-sequence id is stripped explicitly for continuation
    chunks -- otherwise every record would carry a BOS into the middle of the
    context.
    """
    try:
        ids = list(tokenizer.encode(text, add_special_tokens=special))
    except TypeError:
        ids = list(tokenizer.encode(text))

    bos = getattr(tokenizer, "bos_token_id", None)
    if not special and isinstance(bos, int) and ids and ids[0] == bos:
        ids = ids[1:]
    if not ids:
        raise ValueError(f"tokenizer produced no ids for {text!r}")
    return mx.array([ids])


def stop_ids(tokenizer: Any) -> List[int]:
    """Every id that should end a generation.

    mlx-lm's wrapper carries a set in ``eos_token_ids`` (Kimi ends turns with
    ``<|im_end|>``, which is not the bare ``eos_token_id``), so both are used.
    """
    ids = set()
    plural = getattr(tokenizer, "eos_token_ids", None)
    if plural:
        ids.update(int(i) for i in plural)
    for attr in ("eos_token_id", "pad_token_id"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int):
            ids.add(val)
    return sorted(ids)


def preamble_record(tokenizer: Any, text: Optional[str] = None) -> Record:
    return Record(
        key="__preamble__", tokens=encode(tokenizer, text or PREAMBLE, special=True)
    )


def case_records(tokenizer: Any, cases: List[Dict[str, str]]) -> List[Record]:
    return [
        Record(key=c["key"], tokens=encode(tokenizer, c["text"]), text=c["text"])
        for c in cases
    ]


def filler_record(tokenizer: Any, n_tokens: int, key: str = "__filler__") -> Record:
    """Neutral prose truncated to exactly ``n_tokens`` tokens."""
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    text = ""
    ids: List[int] = []
    n = 0
    while len(ids) < n_tokens:
        n += 1
        text += FILLER_SENTENCES.format(n=n)
        ids = list(encode(tokenizer, text)[0].tolist())
    return Record(key=key, tokens=mx.array([ids[:n_tokens]]), text=text)


def probe_secret(
    memory: RecordMemory, tokenizer: Any, case: Dict[str, Any]
) -> ProbeStats:
    """Score a record's whole-body target.

    ``body_stem`` is prompt text the target continues from. Tabular records need
    none, because their target begins at the answer; a note continuation needs
    the verbatim cue it is asked to continue.
    """
    return teacher_forced(
        memory,
        encode(tokenizer, "\n" + case["question"] + case.get("body_stem", "")),
        encode(tokenizer, case["secret"]),
    )


def score_all(
    memory: RecordMemory, tokenizer: Any, cases: List[Dict[str, str]]
) -> Dict[str, ProbeStats]:
    return {c["key"]: probe_secret(memory, tokenizer, c) for c in cases}


def answer_text(
    memory: RecordMemory, tokenizer: Any, question: str, max_tokens: int = 20
) -> str:
    """Greedily answer ``question`` against the memory, leaving it untouched."""
    toks = memory.generate(
        encode(tokenizer, "\n" + question),
        max_tokens=max_tokens,
        stop_ids=stop_ids(tokenizer),
    )
    return tokenizer.decode(toks).strip()


def build_memory(model: Any, records: List[Record]) -> RecordMemory:
    memory = RecordMemory(model)
    memory.ingest_all(records)
    return memory
